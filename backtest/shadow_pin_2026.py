"""The frozen shadow arm's pin: the row-19 models and bet rule it keeps all 2026 season (Phase 37).

WHAT THE PIN IS
---------------
Phase 37 re-fits the LIVE arm's models every four NFL weeks. Beside it, a frozen SHADOW arm keeps
deciding with exactly the models and bet rule that were live when the re-fitting started -- the
row-19 recipe ``recipe-2026-row19-v1`` -- so that at season end the two arms can be compared on the
same games (RFIT-03, RFIT-04, D-04). This module names that frozen arm: its three model ids, its
blend, the converter the blend is bound to, its recipe id, and the chain-fit run record its bet rule
reads, identified by sha256 because the record itself is gitignored generator output.

ONE-WAY BY CONSTRUCTION. Nothing in the codebase writes this module, and the shadow arm reads its
ids from here and NEVER from ``artifacts/latest.json``, which the live re-fit swap rewrites. Stated
plainly because it is easy to forget later in the season: EDITING THIS FILE AFTER THE SHADOW ARM HAS
WRITTEN ITS FIRST ROW DOES NOT FIX A BUG -- IT DESTROYS THE COMPARISON. A frozen arm that moved is
not frozen, and every shadow row already in the ledger would then name a pin that no longer exists.

The id mapping is read-only at run time as well (``types.MappingProxyType``): an item assignment or
``.update()`` on it raises ``TypeError``, so an import cannot repoint the frozen arm in memory.

Every value is COPIED as a literal from the row-19 entry of ``backtest/recipe_registry.py``, never
imported from it, so this record cannot change meaning when the code around it moves. Its witness
(commit and newline-normalized sha256) lands later in ``tests/phase37_state.py``.

Constants only: NO project imports (stdlib ``types`` and ``collections.abc`` only), NO I/O and NO
logic.

ASCII only, no emoji (CLAUDE.md hard constraint).
"""

from __future__ import annotations

from collections.abc import Mapping
from types import MappingProxyType

__all__ = [
    "SHADOW_BLEND",
    "SHADOW_CHAIN_FIT_PATH",
    "SHADOW_CHAIN_FIT_SHA256",
    "SHADOW_CONVERTER",
    "SHADOW_MODEL_IDS",
    "SHADOW_RECIPE",
]


# ---------------------------------------------------------------------------
# The frozen models
# ---------------------------------------------------------------------------

# Model target -> the row-19 model artifact directory. Read-only: assignment raises TypeError.
SHADOW_MODEL_IDS: Mapping[str, str] = MappingProxyType(
    {
        "wp": "wp_20261004_050223",
        "ats": "ats_20261004_050228",
        "ou": "ou_20261004_050232",
    }
)

# The blend tuned on those three models.
SHADOW_BLEND: str = "blend_20261004_050521"

# The market-probability converter that blend is bound to (the WP second test reads its slope).
SHADOW_CONVERTER: str = "market_probability_20260923_195443"


# ---------------------------------------------------------------------------
# The frozen bet rule
# ---------------------------------------------------------------------------

# The registered recipe every shadow row is stamped with.
SHADOW_RECIPE: str = "recipe-2026-row19-v1"

# The tune-only chain-fit run record the row-19 bet rule reads (repo-relative, gitignored).
SHADOW_CHAIN_FIT_PATH: str = "outputs/row19/neutral_hfa_chain_fit.json"

# The sha256 of that record's bytes; the shadow arm refuses a record that does not hash to it.
SHADOW_CHAIN_FIT_SHA256: str = (
    "fab1118b8093bf55fd1cd4d7d8ba80eff3f95b7bd9537c4cf4f3a6b2a6ce2906"
)
