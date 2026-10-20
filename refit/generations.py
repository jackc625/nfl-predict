"""Which re-fit generation decided a ledger row, read from the row's own stamp (D-19, R11, RFIT-08).

WHY A GENERATION IS DERIVED AND NEVER STORED
--------------------------------------------
Phase 37 re-fits the live arm's models every four NFL weeks; each install is a numbered generation,
and the frozen shadow arm's pinned row-19 models are generation 0. Every reader that splits rows by
generation (the A/A record, the arm comparison, the panel) needs to know which models produced a
row. That is already recorded: the stamping site writes each row's ``model_artifact_id``. So the
generation is DERIVED from that stamp here, and no ledger column is added -- ``IMMUTABLE_COLUMNS_V1``
and ``canon_v = 1`` stay untouched, and so does every existing chain hash (must-NOT P6).

THE MAPPING, IN ORDER
---------------------
1. A shadow row is generation 0 ONLY when its id is one of the pin's model ids. Any other id on a
   shadow row -- unknown, NULL or an installed generation's -- is refused by name: the verdict's
   stamp checks read live rows only, so nothing else would catch a corrupted shadow stamp.
2. A live row with a NULL id (weeks 3-4, scored by pre-row-19 models) has no generation and stays
   out of every comparison.
3. A live row with a pinned id is generation 0.
4. A live row with an id of an installed generation maps to that generation's number.
5. Anything else is refused by name (:class:`UnknownGenerationError`), never read as generation 0.

Plan 37-10 extends this module with the installs read from ``ledger/refit/`` and keeps this order.

Batch code: the daily run, the cache build and the CLIs import it; the request path never does.

ASCII only, no emoji (CLAUDE.md hard constraint).
"""

from __future__ import annotations

from collections.abc import Sequence

from forward_ledger.schema import ARM_SHADOW
from forward_ledger.shadow import SHADOW_PIN
from models.artifacts import ResolvedArtifacts

__all__ = [
    "GENERATION_ZERO",
    "UnknownGenerationError",
    "generation_of_row",
    "generation_zero_ids",
]

# The frozen arm's generation: the pinned row-19 models, before any re-fit.
GENERATION_ZERO: int = 0

# The model targets whose ids a row may be stamped with.
_MODEL_TARGETS: tuple[str, ...] = ("wp", "ats", "ou")


class UnknownGenerationError(Exception):
    """A row's stamped model id belongs to no generation it may be counted under (id and arm named).

    Inherits bare ``Exception`` (the ``data.graded_weeks`` rule): a row whose models are unknown
    must never be degraded into generation 0 or quietly dropped.
    """


def generation_zero_ids() -> ResolvedArtifacts:
    """The ids of generation 0: the frozen shadow arm's pin."""
    return SHADOW_PIN.resolved()


def _model_ids(resolved: ResolvedArtifacts) -> frozenset[str]:
    """The three model ids of *resolved* (the blend and converter name no model)."""
    return frozenset(resolved.model_id_for(target) for target in _MODEL_TARGETS)


def generation_of_row(
    model_artifact_id: str | None,
    row_arm: str,
    installs: Sequence[tuple[int, ResolvedArtifacts]] = (),
) -> int | None:
    """The generation whose models produced a row, from the row's stamped model id.

    Args:
        model_artifact_id: The row's stamped ``model_artifact_id`` (None for weeks 3-4).
        row_arm: The row's ``arm`` (``live`` or ``shadow``).
        installs: Each installed generation's number and resolved ids (none before the first re-fit).

    Returns:
        The generation number, or None for a live row with no stamped model id.

    Raises:
        UnknownGenerationError: a shadow row whose id is not a pinned id, or a live row whose id is
            neither pinned nor installed.
    """
    pinned = _model_ids(generation_zero_ids())
    if row_arm == ARM_SHADOW:
        if model_artifact_id in pinned:
            return GENERATION_ZERO
        msg = (
            f"a {row_arm} row is stamped with model id {model_artifact_id!r}, which is not one of "
            f"the frozen pin's ids {sorted(pinned)}; a shadow row is generation 0 only with a "
            "pinned id."
        )
        raise UnknownGenerationError(msg)
    if model_artifact_id is None:
        return None
    if model_artifact_id in pinned:
        return GENERATION_ZERO
    for number, resolved in installs:
        if model_artifact_id in _model_ids(resolved):
            return number
    msg = (
        f"a {row_arm} row is stamped with model id {model_artifact_id!r}, which is neither one of "
        f"the frozen pin's ids {sorted(pinned)} nor an id of any installed generation "
        f"{[number for number, _ in installs]}."
    )
    raise UnknownGenerationError(msg)
