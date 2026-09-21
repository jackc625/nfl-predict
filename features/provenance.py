"""The information-time gate: every value checked against its OWN game's lock (SPEC R2).

Phase 33.2, Plan 33.2-01. This module replaces ``LeakageGate.check_time_fence``, which
compared each row's KICKOFF to the build clock. That asked the wrong question: it blocked
every unplayed game, and it could never catch a value built from post-lock information,
because a kickoff is a schedule fact, not an information time. The question asked here is
the one D33.2-01 poses: was every piece of information behind this value known at or before
18:00 ET on the calendar day before this game's kickoff?

THE SIDECAR, NEVER A COLUMN
---------------------------
A source's information times travel BESIDE its feature frame, as a separate per-row
provenance frame of ``PROVENANCE_COLUMNS``. They are never a column on the feature frame:
``combine_features`` merges every column a source emits (bar a short drop list), so a time
column would reach gold. ``refuse_provenance_columns`` is the dtype guard that makes that a
refusal rather than a hope.

TWO VOCABULARIES, KEPT APART ON PURPOSE
---------------------------------------
``InformationBasis`` describes how a source ESTABLISHES a row's information time
(``per_row`` or ``no_information``). ``SourceCheckState`` describes what the GATE was able
to do with the frame it was handed (``checked`` or ``empty_unchecked``). Merging them would
make ``empty_unchecked`` look like a basis a source could DECLARE -- an exemption mechanism,
which the R2 must-not forbids. A source cannot make itself unchecked; only an empty frame
can, and the gate reports that by name.

ONE LOCK RULE
-------------
Locks come from ``utils.game_lock`` and every comparison goes through
``utils.game_lock.is_admissible``. Both are reached as MODULE ATTRIBUTES at call time
(``_game_lock.lock_frame``, ``_game_lock.is_admissible``), never bound eagerly, so Plan
33.2-02's identity delegate -- which patches ``utils.game_lock`` -- sees every call from
here. An eagerly-bound alias would make that counter silently miss this module.

ASCII only, no emoji (CLAUDE.md hard constraint).
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from datetime import timedelta
from enum import StrEnum
from typing import Any, cast

import pandas as pd

from utils import game_lock as _game_lock

__all__ = [
    "DECLARED_GAME_DURATION",
    "PROVENANCE_COLUMNS",
    "PROVENANCE_NAME_MARKERS",
    "CoverageReport",
    "InformationBasis",
    "InformationTimeGate",
    "InformationTimeViolation",
    "ProvenanceCoverageError",
    "SourceCheckState",
    "UndatedSourceError",
    "build_lock_frame",
    "refuse_provenance_columns",
]

#: THE ONE NAME for the per-row provenance frame's shape (D30-02: one answer on disk).
#: Plans 33.2-12 .. 33.2-17 import this rather than re-spelling three string literals.
PROVENANCE_COLUMNS: tuple[str, str, str] = ("game_id", "basis", "information_time")

#: How long after kickoff a game's RESULT is treated as known. A result-derived value's
#: information time is the END of its latest contributing game: kickoff plus this. Four
#: hours is the duration behind D33.2-01's measurement (0 post-lock inputs at every lock
#: hour tested); Plan 33.2-04's replay re-measures the insensitivity band independently.
DECLARED_GAME_DURATION: timedelta = timedelta(hours=4)

#: Column-name fragments that mark a PROVENANCE column. Any gold column whose lowercased
#: name contains one is refused by ``refuse_provenance_columns``: the sidecar is never a
#: column. ``basis`` is matched as an exact name only (see ``_is_provenance_named``),
#: because as a substring it would be far too broad.
PROVENANCE_NAME_MARKERS: tuple[str, ...] = ("information_time", "provenance")

#: How many offending game ids a refusal names before summarising the rest.
_MAX_NAMED = 10


class InformationTimeViolation(Exception):
    """A value's information post-dates its game's lock, or cannot be established.

    DELIBERATELY AN ``Exception``, NOT A ``ValueError`` OR ``RuntimeError``.
    ``scripts/build_features._SOURCE_LOAD_ERRORS`` catches ``DataIngestionError``,
    ``ValueError``, ``KeyError``, ``TypeError``, ``FileNotFoundError`` and ``OSError`` around
    every source load and turns the failure into an EMPTY frame with a warning. A refusal
    caught there refuses nothing: the build would proceed without the source and exit green,
    which is exactly the silent shape this gate exists to remove. Deriving from ``Exception``
    directly keeps every refusal out of that tuple, and a test asserts it member by member.

    Carries a ``details`` dict so a caller can act on WHICH games failed without parsing the
    message.
    """

    def __init__(self, message: str, details: dict[str, Any] | None = None) -> None:
        super().__init__(message)
        self.details: dict[str, Any] = details or {}


class UndatedSourceError(InformationTimeViolation):
    """A value whose information time cannot be established.

    Raised for a ``per_row`` row with a NULL information time (the ``undated`` refusal R2
    requires), and for a ``no_information`` row whose VALUES contradict the source's
    declared signature -- the claim "this row carries no information" is checked against
    the row, never believed (RESEARCH P2).
    """


class ProvenanceCoverageError(InformationTimeViolation):
    """The provenance frame does not cover the source frame one-to-one, or is malformed.

    Zero-match (the wrong-id-format trap, RESEARCH P8), a duplicate row, a game missing in
    either direction, a game with no lock, the wrong columns, an unknown basis, a
    contradictory ``no_information`` row, or a ``no_information`` row with nothing declared
    to check it against.
    """


class InformationBasis(StrEnum):
    """How a source establishes a row's information time. Exactly two members.

    There is no third member and no ``report_only``: a value is either dated, or declared
    undatable and value-checked. An ``Enum`` (not a dataclass) so its members are readable
    by ``__members__``.
    """

    PER_ROW = "per_row"
    NO_INFORMATION = "no_information"


class SourceCheckState(StrEnum):
    """What the gate was able to do with a source frame. Exactly two members.

    A REPORT vocabulary, separate from ``InformationBasis`` by design (see the module
    docstring). ``EMPTY_UNCHECKED`` is reported, by name, when a source frame has zero rows
    -- which is exactly what a FAILED load produces under ``_SOURCE_LOAD_ERRORS``, and
    which two-way coverage against an empty map would otherwise pass trivially. This plan
    only REPORTS it; Plan 33.2-20 turns it into a build refusal once every source has a
    supplier.
    """

    CHECKED = "checked"
    EMPTY_UNCHECKED = "empty_unchecked"


@dataclass(frozen=True)
class CoverageReport:
    """How much of the build the gate actually covered -- four named sets, never a bare pass.

    A NARROW pass must read as narrow. At Plan 33.2-01 exactly one registry key (``elo``)
    has a supplier, so a report saying only "passed" would read as whole-build coverage.

    Attributes:
        checked_sources: Registry keys the gate checked row by row.
        empty_unchecked_sources: Registry keys whose frame had zero rows, so nothing could
            be checked. Never counted as checked.
        unregistered_sources: Non-empty registry keys whose builder does not yet satisfy
            ``features.protocol.InformationTimeProvider``. Plans 33.2-12 .. 33.2-17 move
            each one into ``checked_sources``.
        post_stage1_sources: Families merged AFTER the Stage-1 loop, which are not registry
            keys at all and cannot be reached by registering a supplier. Kept apart from
            ``unregistered_sources`` so a structural gap is not hidden inside a bookkeeping
            one. Plan 33.2-16 moves the opponent-adjusted family out of this set.
    """

    checked_sources: tuple[str, ...]
    empty_unchecked_sources: tuple[str, ...]
    unregistered_sources: tuple[str, ...]
    post_stage1_sources: tuple[str, ...]


def build_lock_frame(games_df: pd.DataFrame) -> pd.Series:
    """One tz-aware lock per game, from the ONE rule in ``utils.game_lock``.

    A WRAPPER FUNCTION, never an assignment-style re-export: it reaches
    ``utils.game_lock.lock_frame`` as a module attribute at CALL time, so Plan 33.2-02's
    identity delegate (which patches ``utils.game_lock``) counts this call site too.

    Args:
        games_df: The games frame (``game_id`` and ``kickoff_et``).

    Returns:
        ``game_id``-indexed locks.
    """
    return _game_lock.lock_frame(games_df)


def _named(ids: Iterable[object]) -> str:
    """Up to ``_MAX_NAMED`` ids, sorted, with a count of the rest."""
    ordered = sorted(str(i) for i in ids)
    shown = ordered[:_MAX_NAMED]
    more = f" (+{len(ordered) - len(shown)} more)" if len(ordered) > len(shown) else ""
    return f"{shown}{more}"


def _is_null(value: Any) -> bool:
    """True for a scalar null (None, NaN, NaT)."""
    if value is None:
        return True
    try:
        return bool(pd.isna(value))
    except (TypeError, ValueError):
        return False


class InformationTimeGate:
    """The one comparison: each source's per-row information time against its game's lock.

    Stateful across one build: it accumulates which sources it CHECKED and which arrived
    EMPTY, so the build can report both by name. Instantiate one per build.
    """

    def __init__(self) -> None:
        self._checked: list[str] = []
        self._empty: list[str] = []

    @property
    def checked_sources(self) -> tuple[str, ...]:
        """Sources checked row by row, in the order they were checked."""
        return tuple(self._checked)

    @property
    def empty_unchecked_sources(self) -> tuple[str, ...]:
        """Sources whose frame had zero rows. Never included in ``checked_sources``."""
        return tuple(self._empty)

    def check(
        self,
        source_name: str,
        source_frame: pd.DataFrame,
        provenance: pd.DataFrame,
        lock_frame: pd.Series,
        *,
        no_information_signature: Mapping[str, float | None],
    ) -> SourceCheckState:
        """Check one source's provenance against the per-game lock frame.

        There is NO source-wide basis parameter: ``basis`` is a column, because a source's
        rows do not share one (Elo's 2002 week 1 is undatable; week 2 is dated).

        Args:
            source_name: The ``feature_sources`` registry key, named in every refusal.
            source_frame: The source's feature frame (must carry ``game_id``).
            provenance: The source's per-row provenance frame (``PROVENANCE_COLUMNS``).
            lock_frame: ``game_id`` -> tz-aware lock, from :func:`build_lock_frame`.
            no_information_signature: The source's declared neutral values for undatable
                rows (``InformationTimeProvider.no_information_signature()``).

        Returns:
            ``SourceCheckState.EMPTY_UNCHECKED`` for a zero-row source frame, otherwise
            ``SourceCheckState.CHECKED`` once every row has passed.

        Raises:
            ProvenanceCoverageError: malformed frame, coverage gap in either direction,
                zero-match, duplicate, missing lock, contradictory or unsupported
                ``no_information`` row.
            UndatedSourceError: a ``per_row`` row with no time, or a ``no_information`` row
                whose values contradict the declared signature.
            InformationTimeViolation: a ``per_row`` information time after its game's lock,
                or a naive (tz-unaware) information time.
        """
        # 1. An empty frame is reported by name BEFORE anything else. Two-way coverage
        #    between an empty frame and an empty map passes trivially, and an empty frame
        #    is exactly what a failed load becomes -- so without this state a failure
        #    would count as checked.
        if len(source_frame) == 0:
            self._empty.append(source_name)
            return SourceCheckState.EMPTY_UNCHECKED

        self._validate_shape(source_name, source_frame, provenance)
        self._validate_coverage(source_name, source_frame, provenance, lock_frame)

        per_row = provenance.loc[provenance["basis"] == InformationBasis.PER_ROW.value]
        no_info = provenance.loc[
            provenance["basis"] == InformationBasis.NO_INFORMATION.value
        ]

        self._check_per_row(source_name, per_row, lock_frame)
        self._check_no_information(
            source_name, source_frame, no_info, no_information_signature
        )

        self._checked.append(source_name)
        return SourceCheckState.CHECKED

    # -- 2. Shape ---------------------------------------------------------------

    @staticmethod
    def _validate_shape(
        source_name: str, source_frame: pd.DataFrame, provenance: pd.DataFrame
    ) -> None:
        if "game_id" not in source_frame.columns:
            msg = (
                f"source {source_name!r} carries no 'game_id' column, so its rows cannot "
                "be matched to a provenance row or a lock"
            )
            raise ProvenanceCoverageError(msg, {"source": source_name})

        columns = list(provenance.columns)
        if sorted(columns) != sorted(PROVENANCE_COLUMNS) or len(columns) != len(
            PROVENANCE_COLUMNS
        ):
            msg = (
                f"provenance for {source_name!r} must carry exactly the columns "
                f"{list(PROVENANCE_COLUMNS)}; got {columns}"
            )
            raise ProvenanceCoverageError(msg, {"source": source_name})

        allowed = {m.value for m in InformationBasis}
        bases = provenance["basis"]
        unknown = sorted(
            {str(b) for b in bases if _is_null(b) or str(b) not in allowed}
        )
        if unknown:
            msg = (
                f"provenance for {source_name!r} carries basis value(s) {unknown}; the "
                f"only bases are {sorted(allowed)}. There is no third basis and no "
                "report-only mode."
            )
            raise ProvenanceCoverageError(msg, {"source": source_name})

        duplicated = provenance.loc[provenance["game_id"].duplicated(), "game_id"]
        if len(duplicated) > 0:
            msg = (
                f"provenance for {source_name!r} has more than one row for game(s) "
                f"{_named(set(duplicated))}; exactly one row per game is required"
            )
            raise ProvenanceCoverageError(
                msg,
                {"source": source_name, "game_ids": sorted(map(str, set(duplicated)))},
            )

    # -- 3. Two-way coverage, and every game has a lock ------------------------------

    @staticmethod
    def _validate_coverage(
        source_name: str,
        source_frame: pd.DataFrame,
        provenance: pd.DataFrame,
        lock_frame: pd.Series,
    ) -> None:
        source_ids = {str(g) for g in source_frame["game_id"]}
        map_ids = {str(g) for g in provenance["game_id"]}

        if not source_ids & map_ids:
            msg = (
                f"provenance for {source_name!r} matches ZERO of the source frame's "
                f"{len(source_ids)} game ids. This is the wrong-id-format trap (e.g. "
                "'2023_01_ARI_WAS' against '2023_W01_ARI@WAS'): a map keyed in another "
                f"format checks nothing. Source ids e.g. {_named(source_ids)}; map ids "
                f"e.g. {_named(map_ids)}."
            )
            raise ProvenanceCoverageError(msg, {"source": source_name})

        missing_in_map = source_ids - map_ids
        if missing_in_map:
            msg = (
                f"coverage gap (source -> map) in {source_name!r}: "
                f"{len(missing_in_map)} game(s) in the source frame have no provenance "
                f"row: {_named(missing_in_map)}"
            )
            raise ProvenanceCoverageError(
                msg, {"source": source_name, "game_ids": sorted(missing_in_map)}
            )

        missing_in_source = map_ids - source_ids
        if missing_in_source:
            msg = (
                f"coverage gap (map -> source) in {source_name!r}: "
                f"{len(missing_in_source)} provenance row(s) name games the source frame "
                f"does not carry: {_named(missing_in_source)}"
            )
            raise ProvenanceCoverageError(
                msg, {"source": source_name, "game_ids": sorted(missing_in_source)}
            )

        lockless = map_ids - {str(g) for g in lock_frame.index}
        if lockless:
            msg = (
                f"{len(lockless)} game(s) in {source_name!r} have no lock in the build's "
                f"lock frame: {_named(lockless)}. A game with no lock cannot be checked."
            )
            raise ProvenanceCoverageError(
                msg, {"source": source_name, "game_ids": sorted(lockless)}
            )

    # -- 4. per_row: compared to the lock -------------------------------------------

    @staticmethod
    def _check_per_row(
        source_name: str, per_row: pd.DataFrame, lock_frame: pd.Series
    ) -> None:
        undated = [
            str(gid)
            for gid, when in zip(
                per_row["game_id"], per_row["information_time"], strict=True
            )
            if _is_null(when)
        ]
        if undated:
            msg = (
                f"{len(undated)} per_row row(s) in {source_name!r} carry NO information "
                f"time: {_named(undated)}. A dated basis with no date is undated, and an "
                "undated value is refused rather than assumed early."
            )
            raise UndatedSourceError(
                msg, {"source": source_name, "game_ids": sorted(undated)}
            )

        # The timezone discipline carried over from the fence this gate replaces
        # (features/validation.py, CR-01):
        #
        #     This previously read `as_of_ts.tz_localize(col_values.dt.tz)`, which STAMPS
        #     the cutoff's wall clock with the column's zone instead of converting it. A
        #     naive 18:05 ET cutoff became 18:05Z == 14:05 ET, four hours early ...
        #     Localize to ET first, then convert, so the instant is preserved whatever
        #     the host.
        #
        # Under D33.2-01 a naive value RAISES instead of being aligned: the discipline --
        # an instant is converted, never relabelled -- is kept, and the silent alignment
        # branch is removed. ``utils.game_lock.is_admissible`` routes both operands through
        # the project's one strict parser, which refuses a naive value.
        late_ids: list[str] = []
        late_times: list[str] = []
        late_locks: list[str] = []
        for gid, when in zip(
            per_row["game_id"], per_row["information_time"], strict=True
        ):
            lock = lock_frame[str(gid)]
            try:
                admissible = _game_lock.is_admissible(when, lock)
            except ValueError as exc:
                msg = (
                    f"the information time {when!r} for game {gid} in {source_name!r} "
                    "cannot be compared to its lock: it carries no timezone (or does not "
                    "parse). A naive instant is refused, never relabelled."
                )
                raise InformationTimeViolation(
                    msg,
                    {
                        "source": source_name,
                        "game_ids": [str(gid)],
                        "violation_type": "naive_information_time",
                    },
                ) from exc
            if not admissible:
                late_ids.append(str(gid))
                late_times.append(str(pd.Timestamp(cast(Any, when))))
                late_locks.append(str(pd.Timestamp(cast(Any, lock))))

        if late_ids:
            pairs = [
                f"{g}: information_time {t} > lock {lk}"
                for g, t, lk in list(
                    zip(late_ids, late_times, late_locks, strict=True)
                )[:_MAX_NAMED]
            ]
            msg = (
                f"information-time violation in {source_name!r}: {len(late_ids)} value(s) "
                f"were built from information known AFTER their game's lock (18:00 ET the "
                f"day before kickoff): {pairs}"
            )
            raise InformationTimeViolation(
                msg,
                {
                    "source": source_name,
                    "game_ids": late_ids,
                    "information_times": late_times,
                    "locks": late_locks,
                    "violation_type": "information_time",
                },
            )

    # -- 5. no_information: never lock-compared, always value-checked -----------------

    @staticmethod
    def _check_no_information(
        source_name: str,
        source_frame: pd.DataFrame,
        no_info: pd.DataFrame,
        signature: Mapping[str, float | None],
    ) -> None:
        if len(no_info) == 0:
            return

        dated = [
            str(gid)
            for gid, when in zip(
                no_info["game_id"], no_info["information_time"], strict=True
            )
            if not _is_null(when)
        ]
        if dated:
            msg = (
                f"{len(dated)} no_information row(s) in {source_name!r} ALSO carry an "
                f"information time: {_named(dated)}. The declaration contradicts itself "
                "and is refused rather than reconciled."
            )
            raise ProvenanceCoverageError(
                msg, {"source": source_name, "game_ids": sorted(dated)}
            )

        # THE ANTI-EXEMPTION GUARD, and the whole reason the signature is a parameter
        # rather than an assumption: with an empty signature, "no_information" plus "{}"
        # would be a working way for a source to exempt its own rows.
        if not signature:
            msg = (
                f"{source_name!r} declares {len(no_info)} no_information row(s) but an "
                "EMPTY no_information_signature, so the claim cannot be checked against "
                "the rows. A source may not declare rows uncheckable and supply nothing "
                "to check them against."
            )
            raise ProvenanceCoverageError(msg, {"source": source_name})

        absent = sorted(set(signature) - set(source_frame.columns))
        if absent:
            msg = (
                f"{source_name!r}'s no_information_signature names column(s) {absent} "
                "that the source frame does not carry, so the declaration cannot be "
                "checked"
            )
            raise ProvenanceCoverageError(msg, {"source": source_name})

        wanted = {str(g) for g in no_info["game_id"]}
        rows = source_frame.loc[
            source_frame["game_id"].astype(str).isin(sorted(wanted))
        ]
        for column, declared in signature.items():
            for gid, value in zip(rows["game_id"], rows[column], strict=True):
                if declared is None:
                    contradicts = not _is_null(value)
                else:
                    contradicts = _is_null(value) or value != declared
                if contradicts:
                    expected = "null" if declared is None else repr(declared)
                    msg = (
                        f"{source_name!r} declares game {gid} no_information, but its "
                        f"{column!r} is {value!r} where the declared signature requires "
                        f"{expected}. The claim is checked, never believed."
                    )
                    raise UndatedSourceError(
                        msg,
                        {
                            "source": source_name,
                            "game_ids": [str(gid)],
                            "column": column,
                            "value": repr(value),
                            "declared": expected,
                        },
                    )


def _is_provenance_named(column: str) -> bool:
    lowered = column.lower()
    if lowered == "basis":
        return True
    return any(marker in lowered for marker in PROVENANCE_NAME_MARKERS)


def refuse_provenance_columns(
    frame: pd.DataFrame, *, build_clock_columns: tuple[str, ...]
) -> None:
    """The dtype guard: no gold column is a time, and none carries a provenance name.

    The sidecar is never a column. A datetime-typed column in a feature matrix is either an
    information time that escaped the sidecar or a schedule instant a model could key on,
    and both are refused here, before any write.

    ``build_clock_columns`` is the REGISTERED build clock -- the caller passes
    ``scripts.fingerprint_gold.BUILD_CLOCK_COLUMNS``, the one registry of it, rather than a
    literal. It records WHEN a build ran, measures nothing about any game, and no model
    reads it (``models/temporal.py`` excludes it). It is admitted by that registered name
    only; any other datetime column is refused.

    Args:
        frame: A combined or per-target feature matrix.
        build_clock_columns: The registered build-clock column names.

    Raises:
        InformationTimeViolation: naming every datetime-typed or provenance-named column.
    """
    datetime_columns = [
        str(c)
        for c in frame.columns
        if pd.api.types.is_datetime64_any_dtype(frame[c])
        and str(c) not in build_clock_columns
    ]
    provenance_named = [str(c) for c in frame.columns if _is_provenance_named(str(c))]
    if datetime_columns or provenance_named:
        msg = (
            "a feature matrix carries time or provenance columns, which must never reach "
            f"gold (the sidecar is never a column): datetime-typed {datetime_columns}, "
            f"provenance-named {provenance_named}"
        )
        raise InformationTimeViolation(
            msg,
            {
                "violation_type": "provenance_column",
                "datetime_columns": datetime_columns,
                "provenance_named_columns": provenance_named,
            },
        )
