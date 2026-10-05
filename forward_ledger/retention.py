"""Which artifact directories the ledger's replay depends on (Phase 34, D-11, LDGR-09).

A ledger row is replayable only while every artifact that scored it still exists. The ledger
keeps its own copies (``ledger/artifacts/<id>/``, Plan 34-08), and D-11 adds a second protection:
no tool may delete an artifact directory a ledger row references. :func:`referenced_artifact_ids`
is the one answer to "which ids are those?":

  * every non-NULL ``model_artifact_id`` and ``blend_id`` of a row entry (pre-verdict rows carry
    NULL stamps and reference nothing);
  * every directory the ledger keeps a copy of -- which includes the converter each blend is bound
    to, an id no row column names (34-RESEARCH Pitfall 16).

ABSENT IS NOT UNREADABLE. No ledger yet references nothing, so an absent store is an empty set.
A store that exists but cannot be read RAISES (``LedgerFormatError``): reading it as "nothing
referenced" would license deleting exactly the artifacts it protects.

ASCII only, no emoji (CLAUDE.md hard constraint).
"""

from __future__ import annotations

from pathlib import Path

from forward_ledger.artifacts_copy import ARTIFACT_COPIES_DIRNAME
from forward_ledger.canonical import ENTRY_KIND_ROW
from forward_ledger.store import LEDGER_DIR, read_entries

__all__ = ["REFERENCE_COLUMNS", "referenced_artifact_ids"]

# The row columns that name an artifact directory.
REFERENCE_COLUMNS: tuple[str, ...] = ("model_artifact_id", "blend_id")


def referenced_artifact_ids(ledger_dir: Path | str = LEDGER_DIR) -> frozenset[str]:
    """Every artifact id a ledger row references or the ledger keeps a copy of.

    Args:
        ledger_dir: The ledger directory (``ledger`` in production).

    Returns:
        The ids; empty when the ledger does not exist yet.

    Raises:
        LedgerFormatError: the store exists but cannot be read.
    """
    directory = Path(ledger_dir)
    ids = {
        entry.immutable[column]
        for entry in read_entries(directory)
        if entry.kind == ENTRY_KIND_ROW
        for column in REFERENCE_COLUMNS
        if entry.immutable.get(column) is not None
    }
    copies = directory / ARTIFACT_COPIES_DIRNAME
    if copies.is_dir():
        # A ".staging-*" directory is a copy still being written, not a reference.
        ids |= {
            child.name
            for child in copies.iterdir()
            if child.is_dir() and not child.name.startswith(".")
        }
    return frozenset(ids)
