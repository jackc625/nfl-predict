"""Output-path guards shared by the tools that must never write under ``data/``.

Phase 30's hard boundary is that nothing writes under ``data/`` outside the ONE
sanctioned, fingerprinted gold rebuild (SPEC R1, T-30-14). Three tools state that
prohibition in prose; ``backtest.group_gate`` was the only one that ENFORCED it, and its
docstring explains why the check has to run BEFORE the work: "a refusal that arrived
after twelve walk-forward re-fits would be a refusal nobody could afford to trust."

The two tools that operate directly on the gold tree -- ``scripts/fingerprint_gold.py``
and ``scripts/resync_games_duckdb.py`` -- were the two that did not adopt it, and the
phase's hard-boundary hash manifest reads through ``load_dataframe``, so it would not
have caught such a write either (WR-07).

This module holds the ONE implementation. A second copy of a rule is free to drift away
from the rule everything else applies, which is the failure mode this project already
names in ``scripts/fingerprint_gold._discrete_indicator_predicate``.
"""

from __future__ import annotations

from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent.parent

DEFAULT_OUTPUT_SUGGESTION = "outputs/"


def reject_data_path(
    path: Path | str,
    *,
    what: str = "this output",
    suggestion: str = DEFAULT_OUTPUT_SUGGESTION,
) -> Path:
    """Resolve *path*, REFUSING anything under a ``data/`` tree.

    Both this repository's ``data/`` and the current working directory's ``data/`` are
    rejected, because a tool can legitimately be run from elsewhere and the operator's
    intent -- "do not put run output in the data lake" -- is the same either way.

    Args:
        path: The requested output path.
        what: A short noun phrase naming what would have been written, used in the
            refusal message so the operator knows which tool refused.
        suggestion: Where the output SHOULD go, quoted in the message. A refusal that
            does not say what to do instead just gets worked around.

    Returns:
        The resolved absolute path.

    Raises:
        ValueError: If the path lands under either ``data/`` tree.
    """
    resolved = Path(path).expanduser().resolve()
    for root in (_REPO_ROOT / "data", Path.cwd() / "data"):
        try:
            resolved.relative_to(root.resolve())
        except ValueError:
            continue
        msg = (
            f"Refusing to write {what} to '{path}'. Nothing in Phase 30 writes under "
            "data/ outside the ONE sanctioned, fingerprinted gold rebuild (SPEC R1, "
            f"T-30-14). Write it under '{suggestion}' instead."
        )
        raise ValueError(msg)
    return resolved
