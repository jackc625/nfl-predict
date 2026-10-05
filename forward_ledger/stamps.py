"""The ONE site that stamps a ledger row's Phase-34 immutable columns (LDGR-03, LDGR-09, LDGR-11).

WHAT IS STAMPED, AND FROM WHERE
-------------------------------
Eleven immutable columns follow ``decided_at_utc`` in the v1 schema, and :func:`stamp_ledger_rows`
is the only production code that sets them (an AST scan in
``tests/unit/test_ledger_stamping_site.py`` holds that; the migration may only set them NULL):

  * ``arm`` -- ``live`` for every row this phase writes (Phase 37 writes ``shadow``);
  * ``model_artifact_id`` / ``blend_id`` -- from the ids the decision was SCORED by, resolved once
    (Plan 34-05), never re-read here, so a ``latest.json`` swap mid-run cannot mis-stamp a row;
  * ``recipe_id`` / ``fill_convention_id`` -- the in-force registry entry and ``fill-v1``;
  * ``upstream_capture_key`` / ``gold_generation_key`` / ``odds_snapshot_digest`` -- the
    reproduction key (``forward_ledger.repro_key``);
  * ``decision_snapshot_digest`` -- the stored decision-input snapshot (``forward_ledger.snapshots``);
  * ``verdict_scope`` -- ``verdict`` only inside a declared scope, else ``pre_verdict``;
  * ``regime_label`` -- ``bootstrap_regime`` on weeks 2-4 (D40-08), else NULL.

It never touches ``decided_at_utc``, ``provenance`` or ``validation_type``: those are set by the
decision's own emission and stamping paths.

THE REFUSALS
------------
  * no resolved ids (the bundle found no ``latest.json``) -> ``MissingStampError``: a stamp is
    never invented;
  * scoring ids the in-force recipe entry does not list -> :class:`RecipeArtifactMismatchError`: a
    row may only claim a recipe that actually produced its models (T-34-27);
  * an unregistered recipe or fill id -> the declarations module's named refusals;
  * no verdict-scope declaration while the ledger already holds verdict rows ->
    ``VerdictScopeUndeclaredError``. Before the first verdict row a missing declaration honestly
    means "pre-verdict"; after it, a missing declaration means the declaration was lost, and
    labelling the new row pre-verdict would silently drop it from the count (34-RESEARCH
    Pitfall 13).

THE RUNTIME POINTER
-------------------
:data:`IN_FORCE_RECIPE_ID` names the registry entry new rows are stamped with. Phase 37 moves this
pointer when it registers a new recipe; the registry record itself is never edited.

ASCII only, no emoji (CLAUDE.md hard constraint).
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from backtest.fill_conventions import FILL_CONVENTION_ID
from forward_ledger.declarations import (
    BOOTSTRAP_REGIME_WEEKS,
    VerdictScope,
    VerdictScopeUndeclaredError,
    resolve_fill_convention,
    resolve_recipe,
    verdict_scope_label,
)
from forward_ledger.schema import ARM_LIVE, ARMS, REGIME_LABEL_BOOTSTRAP
from forward_ledger.store import InvalidArmError, MissingStampError

if TYPE_CHECKING:
    import pandas as pd

    from forward_ledger.repro_key import ReproKey
    from models.artifacts import ResolvedArtifacts

__all__ = [
    "IN_FORCE_RECIPE_ID",
    "RecipeArtifactMismatchError",
    "stamp_ledger_rows",
]

IN_FORCE_RECIPE_ID = "recipe-2026-row19-v1"


class RecipeArtifactMismatchError(Exception):
    """The decision was scored by artifact ids the in-force recipe entry does not list.

    Inherits bare ``Exception`` (the ``data.graded_weeks`` rule): a row whose recipe did not
    produce its models must never be written, and a broad handler must not turn this into "skip".
    """


def _require_recipe_artifacts(recipe_id: str, resolved: ResolvedArtifacts) -> None:
    """Refuse unless every id that scored the decision is the recipe entry's own."""
    entry = resolve_recipe(recipe_id)
    expected = {
        "wp": entry.model_artifact_ids.get("wp"),
        "ats": entry.model_artifact_ids.get("ats"),
        "ou": entry.model_artifact_ids.get("ou"),
        "blend": entry.blend_id,
        "converter": entry.converter_id,
    }
    actual = {
        "wp": resolved.wp,
        "ats": resolved.ats,
        "ou": resolved.ou,
        "blend": resolved.blend,
        "converter": resolved.converter,
    }
    mismatched = [
        f"{name} {actual[name]!r} (the recipe lists {expected[name]!r})"
        for name in expected
        if actual[name] != expected[name]
    ]
    if mismatched:
        msg = (
            f"the decision was scored by artifacts the recipe {recipe_id!r} does not list: "
            + "; ".join(mismatched)
            + ". A row may only claim the recipe that produced its models."
        )
        raise RecipeArtifactMismatchError(msg)


def _require_value(name: str, value: str | None) -> None:
    if not value:
        msg = f"no {name} to stamp; a ledger row is attributed when it is decided, never later"
        raise MissingStampError(msg)


def stamp_ledger_rows(
    frame: pd.DataFrame,
    *,
    resolved: ResolvedArtifacts | None,
    repro: ReproKey,
    snapshot_digest: str,
    scope: VerdictScope | None,
    ledger_has_verdict_rows: bool,
    recipe_id: str = IN_FORCE_RECIPE_ID,
    fill_convention_id: str = FILL_CONVENTION_ID,
    arm: str = ARM_LIVE,
) -> pd.DataFrame:
    """A copy of *frame* with the 11 Phase-34 immutable columns set on every row.

    Args:
        frame: Decided bet-list rows (``game_id``, ``season``, ``week``, ``target`` at least).
        resolved: The ids the decision was scored by; None is refused.
        repro: The decision's reproduction key.
        snapshot_digest: The digest of its stored decision-input snapshot.
        scope: The committed verdict-scope declaration, or None when none exists.
        ledger_has_verdict_rows: Whether the ledger already holds any ``verdict`` row.
        recipe_id: The registry entry to stamp (the in-force pointer by default).
        fill_convention_id: The fill convention to stamp (``fill-v1`` by default).
        arm: ``live`` (Phase 34) or ``shadow`` (Phase 37).

    Returns:
        The stamped copy; *frame* is not modified.

    Raises:
        MissingStampError: *resolved* is None, or a reproduction value or the snapshot digest
            is empty.
        InvalidArmError: *arm* is outside ``ARMS``.
        UnknownRecipeError, UnknownFillConventionError: an id does not resolve.
        RecipeArtifactMismatchError: a scoring id is not the recipe entry's own.
        VerdictScopeUndeclaredError: no declaration although verdict rows already exist.
    """
    if resolved is None:
        msg = (
            "the decision carries no resolved artifact ids (no latest.json was read), so its "
            "model and blend stamps cannot be taken from scoring; a stamp is never invented"
        )
        raise MissingStampError(msg)
    if arm not in ARMS:
        msg = f"arm {arm!r} is outside the closed vocabulary {ARMS}"
        raise InvalidArmError(msg)
    _require_value("upstream capture key", repro.upstream_capture_key)
    _require_value("gold generation key", repro.gold_generation_key)
    _require_value("odds snapshot digest", repro.odds_snapshot_digest)
    _require_value("decision snapshot digest", snapshot_digest)
    _require_recipe_artifacts(recipe_id, resolved)
    resolve_fill_convention(fill_convention_id)
    if scope is None and ledger_has_verdict_rows:
        msg = (
            "the ledger already holds verdict rows but no verdict-scope declaration can be "
            "loaded; a new row is refused rather than silently labelled pre_verdict (LDGR-10)"
        )
        raise VerdictScopeUndeclaredError(msg)

    stamped = frame.copy()
    seasons = [int(season) for season in stamped["season"]]
    weeks = [int(week) for week in stamped["week"]]

    stamped["arm"] = arm
    stamped["model_artifact_id"] = [
        resolved.model_id_for(str(target)) for target in stamped["target"]
    ]
    stamped["blend_id"] = resolved.blend
    stamped["recipe_id"] = recipe_id
    stamped["fill_convention_id"] = fill_convention_id
    stamped["upstream_capture_key"] = repro.upstream_capture_key
    stamped["gold_generation_key"] = repro.gold_generation_key
    stamped["odds_snapshot_digest"] = repro.odds_snapshot_digest
    stamped["decision_snapshot_digest"] = snapshot_digest
    stamped["verdict_scope"] = [
        verdict_scope_label(season, week, scope)
        for season, week in zip(seasons, weeks, strict=True)
    ]
    stamped["regime_label"] = [
        REGIME_LABEL_BOOTSTRAP if week in BOOTSTRAP_REGIME_WEEKS else None
        for week in weeks
    ]
    return stamped
