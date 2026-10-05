"""The ONE loader for the pre-registration records a ledger row is stamped against (Phase 34).

WHAT IT RESOLVES
----------------
Three committed records, each refused by name when it does not resolve:

  * the 2026 verdict-scope declaration (LDGR-10, D-15) -- ``backtest/verdict_scope_2026.py``,
    committed by Plan 34-22 at go-live, BEFORE week W's first lock. Until then it does not exist,
    and :func:`load_verdict_scope` says so with :class:`VerdictScopeUndeclaredError`. A missing
    declaration is NEVER read as a default scope: a verdict computed against a scope nobody
    declared is the after-the-fact rule choice the pre-registration exists to forbid;
  * the recipe registry (LDGR-03) -- ``backtest.recipe_registry.RECIPE_REGISTRY``;
  * the fill conventions (LDGR-11) -- ``backtest.fill_conventions.REGISTERED_FILL_CONVENTIONS``.

WHY THE DECLARATION PINS NO RECIPE SET
--------------------------------------
Phase 37 registers new recipes mid-season by pre-registration (RFIT-01). A verdict row's recipe
only has to resolve in the committed registry; pinning the set here would make every honest
Phase-37 re-fit look like a scope violation.

WHEN A ROW LEARNS ITS LABEL (34-RESEARCH Pitfall 13)
----------------------------------------------------
:func:`verdict_scope_label` is ``verdict`` only when a declaration EXISTS, the season matches and
``start_week <= week <= end_week``; every other case -- including "no declaration yet" -- is
``pre_verdict``. A row is therefore never counted on a scope that was not on disk when it was
written.

THE NAMED REFUSALS
------------------
:class:`VerdictScopeUndeclaredError`, :class:`UnknownRecipeError` and
:class:`UnknownFillConventionError` inherit bare ``Exception``, the ``data.graded_weeks`` rule:
several loaders in this repository catch ``ValueError`` / ``RuntimeError`` broadly and degrade to
an empty result, and none of these refusals may be converted into "nothing declared".
:class:`VerdictScopeMalformedError` is a ``ValueError`` by the plan's contract: it is a defect in a
committed file, reported with the missing constant's name.

IMPORT POSITION
---------------
The two ``backtest`` records are imported INSIDE the resolving functions, the idiom
``data/graded_weeks.py`` uses, so importing this module does not pull them in.

ASCII only, no emoji (CLAUDE.md hard constraint).
"""

from __future__ import annotations

import importlib
from dataclasses import dataclass
from types import ModuleType
from typing import TYPE_CHECKING

from forward_ledger.schema import VERDICT_SCOPE_PRE_VERDICT, VERDICT_SCOPE_VERDICT

if TYPE_CHECKING:
    from backtest.recipe_registry import RecipeEntry

__all__ = [
    "BOOTSTRAP_REGIME_WEEKS",
    "REQUIRED_DECLARATION_CONSTANTS",
    "VERDICT_SCOPE_MODULE",
    "UnknownFillConventionError",
    "UnknownRecipeError",
    "VerdictScope",
    "VerdictScopeMalformedError",
    "VerdictScopeUndeclaredError",
    "load_verdict_scope",
    "resolve_fill_convention",
    "resolve_recipe",
    "verdict_scope_label",
]

# D40-08: weeks 2-4 were decided under the cold-start bootstrap regime and carry
# ``regime_label = bootstrap_regime``. Playoff weeks and neutral-site games are NOT excluded here;
# they count (D40-08), which the declaration records.
BOOTSTRAP_REGIME_WEEKS: tuple[int, ...] = (2, 3, 4)

# The module Plan 34-22 commits at go-live. It does not exist before then.
VERDICT_SCOPE_MODULE: str = "backtest.verdict_scope_2026"

# The declaration's constants, each REQUIRED, in the order they map onto :class:`VerdictScope`.
REQUIRED_DECLARATION_CONSTANTS: tuple[str, ...] = (
    "VERDICT_SEASON",
    "VERDICT_START_WEEK",
    "VERDICT_END_WEEK",
    "INCLUDES_PLAYOFF_WEEKS",
    "INCLUDES_NEUTRAL_SITE_GAMES",
    "COUNTED_ARM",
    "OUTCOME_RULE",
    "FILL_CONVENTION_ID",
    "BOOTSTRAP_REGIME_WEEKS",
)


class VerdictScopeUndeclaredError(Exception):
    """The verdict-scope declaration module does not exist. NOT a claim of an empty scope."""


class VerdictScopeMalformedError(ValueError):
    """The declaration module exists but lacks a required constant (named in the message)."""


class UnknownRecipeError(Exception):
    """A ``recipe_id`` that is missing or absent from the committed recipe registry."""


class UnknownFillConventionError(Exception):
    """A fill convention id that is missing or not a registered convention."""


@dataclass(frozen=True)
class VerdictScope:
    """The committed 2026 verdict scope, one field per required declaration constant."""

    season: int
    start_week: int
    end_week: int
    includes_playoff_weeks: bool
    includes_neutral_site_games: bool
    counted_arm: str
    outcome_rule: str
    fill_convention_id: str
    bootstrap_regime_weeks: tuple[int, ...]


def _is_absent(error: ModuleNotFoundError, module_name: str) -> bool:
    """True when *error* says *module_name* itself (or a package above it) does not exist.

    A declaration that EXISTS but imports something missing is a broken declaration, not an
    absent one, and must surface as the original error rather than as "undeclared".
    """
    missing = error.name
    return missing is not None and (
        missing == module_name or module_name.startswith(missing + ".")
    )


def load_verdict_scope(module_name: str = VERDICT_SCOPE_MODULE) -> VerdictScope:
    """Load the committed verdict-scope declaration.

    Raises:
        VerdictScopeUndeclaredError: *module_name* does not exist.
        VerdictScopeMalformedError: it exists but lacks one or more required constants.
    """
    try:
        module: ModuleType = importlib.import_module(module_name)
    except ModuleNotFoundError as error:
        if not _is_absent(error, module_name):
            raise
        msg = (
            f"no verdict-scope declaration: module {module_name!r} does not exist. A verdict "
            "is never computed against a default scope; the declaration is committed before "
            "week W's first lock (LDGR-10, D-15)."
        )
        raise VerdictScopeUndeclaredError(msg) from error

    missing = [
        name for name in REQUIRED_DECLARATION_CONSTANTS if not hasattr(module, name)
    ]
    if missing:
        msg = (
            f"the verdict-scope declaration {module_name!r} lacks required constant(s) "
            f"{missing}; it is refused rather than completed with defaults."
        )
        raise VerdictScopeMalformedError(msg)

    return VerdictScope(
        season=module.VERDICT_SEASON,
        start_week=module.VERDICT_START_WEEK,
        end_week=module.VERDICT_END_WEEK,
        includes_playoff_weeks=module.INCLUDES_PLAYOFF_WEEKS,
        includes_neutral_site_games=module.INCLUDES_NEUTRAL_SITE_GAMES,
        counted_arm=module.COUNTED_ARM,
        outcome_rule=module.OUTCOME_RULE,
        fill_convention_id=module.FILL_CONVENTION_ID,
        bootstrap_regime_weeks=tuple(module.BOOTSTRAP_REGIME_WEEKS),
    )


def verdict_scope_label(season: int, week: int, scope: VerdictScope | None) -> str:
    """``verdict`` iff *scope* exists, the season matches and the week is in its range."""
    if (
        scope is not None
        and season == scope.season
        and scope.start_week <= week <= scope.end_week
    ):
        return VERDICT_SCOPE_VERDICT
    return VERDICT_SCOPE_PRE_VERDICT


def resolve_recipe(recipe_id: str | None) -> RecipeEntry:
    """The committed registry entry for *recipe_id*.

    Raises:
        UnknownRecipeError: *recipe_id* is missing or not registered.
    """
    from backtest.recipe_registry import RECIPE_REGISTRY

    if recipe_id is None or recipe_id not in RECIPE_REGISTRY:
        msg = (
            f"recipe_id {recipe_id!r} does not resolve in the committed recipe registry "
            f"{sorted(RECIPE_REGISTRY)}; a row is never stamped with an unregistered recipe."
        )
        raise UnknownRecipeError(msg)
    return RECIPE_REGISTRY[recipe_id]


def resolve_fill_convention(fill_id: str | None) -> str:
    """Return *fill_id* when it is a registered fill convention.

    Raises:
        UnknownFillConventionError: *fill_id* is missing or not registered.
    """
    from backtest.fill_conventions import REGISTERED_FILL_CONVENTIONS

    if fill_id is None or fill_id not in REGISTERED_FILL_CONVENTIONS:
        msg = (
            f"fill convention id {fill_id!r} is not a registered convention "
            f"{REGISTERED_FILL_CONVENTIONS}; a row without a resolvable fill id is refused "
            "(LDGR-11)."
        )
        raise UnknownFillConventionError(msg)
    return fill_id
