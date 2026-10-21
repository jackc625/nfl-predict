"""Which re-fit generation decided a ledger row, and which one is in force (D-19, D-20, R11, RFIT-08).

WHY A GENERATION IS DERIVED AND NEVER STORED
--------------------------------------------
Phase 37 re-fits the live arm's models every four NFL weeks; each install is a numbered generation,
and the frozen shadow arm's pinned row-19 models are generation 0. Every reader that splits rows by
generation (the A/A record, the arm comparison, the panel) needs to know which models produced a
row. That is already recorded: the stamping site writes each row's ``model_artifact_id``. So the
generation is DERIVED from that stamp here, and no ledger column is added -- ``IMMUTABLE_COLUMNS_V1``
and ``canon_v = 1`` stay untouched, and so does every existing chain hash (must-NOT P6).

WHERE THE GENERATIONS COME FROM (ROLLING-REFIT-PREREGISTRATION.md section 3)
-----------------------------------------------------------------------------
:func:`installed_generations` reads ``ledger/refit/dates.jsonl`` (``refit.records``):

* generation 0 is the pin (``forward_ledger.shadow.SHADOW_PIN``), its recipe and chain-fit record;
* generation N is the ids named by the last ``install_intent`` of the N-th cadence date, in date
  order, whose last install intent COUNTS. It counts when the date closed ``ran`` or ``ran late``,
  or when the date is still open and no ``refused`` attempt follows that intent. An ``installed``
  or a ``crashed`` attempt after it, or no attempt at all, does not stop it counting, because a
  crash can come after the swap wrote ``latest.json`` (research Pitfall 2); the only refusal an
  intent's own attempt records after it is raised before the swap writes, so a ``refused`` attempt
  after the intent proves those ids were never served;
* a date closed ``not run`` consumes no number.

Generation N's bet rule is resolved ONLY by ``refit.records.bet_rule_record_path(ledger_dir,
intent["bet_rule_sha256"])`` -- the one join of the stored path convention -- never by joining the
stored ``bet_rule_path`` text.

THE ROW MAPPING, IN ORDER
-------------------------
1. A shadow row is generation 0 ONLY when its id is one of the pin's model ids. Any other id on a
   shadow row -- unknown, NULL or an installed generation's -- is refused by name: the verdict's
   stamp checks read live rows only, so nothing else would catch a corrupted shadow stamp.
2. A live row with a NULL id (weeks 3-4, scored by pre-row-19 models) has no generation and stays
   out of every comparison.
3. A live row with a pinned id is generation 0.
4. A live row with an id of an installed generation maps to that generation's number.
5. Anything else is refused by name (:class:`UnknownGenerationError`), never read as generation 0.

THE GENERATION IN FORCE
-----------------------
:func:`in_force_generation` maps the ids a decision was resolved with (``latest.json``, read once)
to exactly one generation. The pin's ids are generation 0 without reading any record, so a
generation-0 decision depends on nothing it did not depend on before Phase 37 (D-04).

THE WEBSITE'S EDGE BANDS (D-36)
-------------------------------
Record section 12: the display follows the generation in force, "so a game's displayed band is the
band its live bet rule used". Two resolvers serve the cache build, both called OUTSIDE ``api/``
(``api/`` imports no ``refit`` or ``backtest`` module) and handed to ``api.cache.populate_cache`` as
plain mappings:

* :func:`decided_edge_thresholds` -- every game with a live ledger row is banded with the
  thresholds of the generation that DECIDED it, read from the row's stamped model id. Generation 0,
  and a live row with no stamp (weeks 3-4), keep the row-19 constants (None), so an install never
  re-labels a game a live bet rule already decided;
* :func:`in_force_edge_thresholds` -- every other game (the 2021-2025 backtest games, and a game no
  live rule has decided yet) is banded with the in-force generation's thresholds. Generation 0 --
  and an artifacts directory with no ``latest.json`` -- answers None, the row-19 constants.

While generation 0 is in force both answer None for every game, so the bands are exactly as before.

Batch code: the daily run, the cache build and the CLIs import it; the request path never does.

ASCII only, no emoji (CLAUDE.md hard constraint).
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

from backtest.refit_preregistration_2026 import REFIT_RECIPE_ID
from forward_ledger.canonical import ENTRY_KIND_ROW
from forward_ledger.schema import ARM_LIVE, ARM_SHADOW
from forward_ledger.shadow import SHADOW_PIN
from models.artifacts import ResolvedArtifacts, resolve_production_artifacts
from refit.records import (
    RefitRecordFormatError,
    bet_rule_record_path,
    read_date_events,
)

if TYPE_CHECKING:
    from forward_ledger.store import LedgerEntry

__all__ = [
    "GENERATION_ZERO",
    "EdgeThresholds",
    "Generation",
    "UnknownGenerationError",
    "decided_edge_thresholds",
    "generation_for_ids",
    "generation_of_row",
    "generation_zero_ids",
    "in_force_edge_thresholds",
    "in_force_generation",
    "installed_generations",
]

# The frozen arm's generation: the pinned row-19 models, before any re-fit.
GENERATION_ZERO: int = 0

# One generation's edge-band thresholds, ``{target -> (high, medium) | None}`` (D-36).
EdgeThresholds = dict[str, tuple[float, float] | None]

# The model targets whose ids a row may be stamped with.
_MODEL_TARGETS: tuple[str, ...] = ("wp", "ats", "ou")

# The five ids an install intent names.
_INTENT_ID_KEYS: tuple[str, ...] = ("wp", "ats", "ou", "blend", "converter")

# The closed statuses under which a date's last install intent counts, and the one that never does.
_COUNTING_CLOSED_STATUSES: frozenset[str] = frozenset({"ran", "ran late"})
_NOT_RUN: str = "not run"
_REFUSED: str = "refused"


class UnknownGenerationError(Exception):
    """Ids, or a row's stamped model id, that belong to no generation they may be counted under.

    Inherits bare ``Exception`` (the ``data.graded_weeks`` rule): a row or a decision whose models
    are unknown must never be degraded into generation 0 or quietly dropped.
    """


@dataclass(frozen=True)
class Generation:
    """One generation: the ids that decide under it, its recipe and its bet rule.

    Attributes:
        number: 0 for the pin, N for the N-th counted install.
        ids: The five ids (wp, ats, ou, blend, converter) the generation decides with.
        recipe: The recipe id its live rows are stamped with (``recipe-2026-row19-v1`` for
            generation 0, ``recipe-2026-refit-v1`` for every re-fit generation).
        bet_rule_path: The bet-rule record it decides with: the pin's chain-fit record for
            generation 0, ``<ledger>/refit/bet_rules/<sha256>.json`` for generation N.
        date_index: The cadence date's 1-based index (None for generation 0).
        list_week: The cadence date list's A/A week (None for generation 0).
    """

    number: int
    ids: ResolvedArtifacts
    recipe: str
    bet_rule_path: Path
    date_index: int | None
    list_week: int | None


def generation_zero_ids() -> ResolvedArtifacts:
    """The ids of generation 0: the frozen shadow arm's pin."""
    return SHADOW_PIN.resolved()


def _generation_zero() -> Generation:
    return Generation(
        number=GENERATION_ZERO,
        ids=generation_zero_ids(),
        recipe=SHADOW_PIN.recipe,
        bet_rule_path=Path(SHADOW_PIN.chain_fit_path),
        date_index=None,
        list_week=None,
    )


def _intent_ids(intent: Mapping[str, Any]) -> ResolvedArtifacts:
    """The five ids an ``install_intent`` names, refused by name when one is missing."""
    ids = intent["ids"]
    missing = [
        key for key in _INTENT_ID_KEYS if not isinstance(ids, Mapping) or key not in ids
    ]
    if missing:
        msg = (
            f"the install_intent of cadence date {intent['date']} (list week "
            f"{intent['list_week']}, index {intent['date_index']}) names no id for {missing}"
        )
        raise RefitRecordFormatError(msg)
    return ResolvedArtifacts(
        wp=str(ids["wp"]),
        ats=str(ids["ats"]),
        ou=str(ids["ou"]),
        blend=str(ids["blend"]),
        converter=None if ids["converter"] is None else str(ids["converter"]),
    )


def _counting_intent(events: Sequence[Mapping[str, Any]]) -> Mapping[str, Any] | None:
    """The date's last install intent when it counts (section 3's rule), else None."""
    intents = [
        position
        for position, event in enumerate(events)
        if event["event"] == "install_intent"
    ]
    if not intents:
        return None
    last = intents[-1]
    closed = [event["status"] for event in events if event["event"] == "closed"]
    if closed:
        # The first closed event decides the date (refit.cadence.date_statuses).
        return events[last] if closed[0] in _COUNTING_CLOSED_STATUSES else None
    refused_after = any(
        event["event"] == "attempt" and event["outcome"] == _REFUSED
        for event in events[last + 1 :]
    )
    return None if refused_after else events[last]


def installed_generations(
    ledger_dir: Path | str, *, skip_torn_tail: bool = False
) -> tuple[Generation, ...]:
    """Generation 0 and every counted install, in number order (section 3's rule).

    Args:
        skip_torn_tail: Read ``dates.jsonl`` leaving out a torn last line
            (``refit.records.read_date_events``); the live decision path's read only.

    Raises:
        RefitRecordFormatError: a ``dates.jsonl`` line is malformed, or an intent lacks an id.
    """
    by_date: dict[tuple[int, int], list[Mapping[str, Any]]] = {}
    for event in read_date_events(ledger_dir, skip_torn_tail=skip_torn_tail):
        key = (int(event["list_week"]), int(event["date_index"]))
        by_date.setdefault(key, []).append(event)

    ordered = sorted(
        by_date.values(),
        key=lambda events: (str(events[0]["date"]), events[0]["date_index"]),
    )
    generations = [_generation_zero()]
    for events in ordered:
        intent = _counting_intent(events)
        if intent is None:
            continue
        generations.append(
            Generation(
                number=len(generations),
                ids=_intent_ids(intent),
                recipe=REFIT_RECIPE_ID,
                bet_rule_path=bet_rule_record_path(
                    ledger_dir, intent["bet_rule_sha256"]
                ),
                date_index=int(intent["date_index"]),
                list_week=int(intent["list_week"]),
            )
        )
    return tuple(generations)


def generation_for_ids(
    resolved: ResolvedArtifacts, generations: Sequence[Generation]
) -> Generation:
    """The one generation whose five ids are exactly *resolved*.

    Raises:
        UnknownGenerationError: no generation names these ids (all five are named).
    """
    for generation in generations:
        if generation.ids == resolved:
            return generation
    msg = (
        f"the ids wp={resolved.wp!r}, ats={resolved.ats!r}, ou={resolved.ou!r}, "
        f"blend={resolved.blend!r}, converter={resolved.converter!r} are neither the frozen "
        "pin's nor any installed generation's "
        f"{[generation.number for generation in generations]}; a decision is never stamped "
        "under a generation that did not produce its models."
    )
    raise UnknownGenerationError(msg)


def in_force_generation(
    resolved: ResolvedArtifacts, ledger_dir: Path | str
) -> Generation:
    """The generation the ids *resolved* (read once from ``latest.json``) decide under.

    The pin's ids are generation 0 without reading any record (D-04). A torn last line of
    ``dates.jsonl`` is left out and logged (WR-A02): a crash mid-append on the re-fit side never
    stands between a game and its lock (D-15); damage on any other line still raises.

    Raises:
        UnknownGenerationError: the ids are neither pinned nor an installed generation's.
        RefitRecordFormatError: the generation records cannot be read.
    """
    zero = _generation_zero()
    if resolved == zero.ids:
        return zero
    return generation_for_ids(
        resolved, installed_generations(ledger_dir, skip_torn_tail=True)
    )


def _generation_edge_thresholds(generation: Generation) -> EdgeThresholds | None:
    """*generation*'s bet-rule thresholds; None for generation 0 (the row-19 constants)."""
    if generation.number == GENERATION_ZERO:
        return None
    # Imported here, not at load time: backtest.weekly_bet_list imports api.cache and the
    # selection stack, which the generation-0 answer above never needs.
    from backtest.weekly_bet_list import load_bet_rule_record

    _fits, thresholds = load_bet_rule_record(generation.bet_rule_path)
    return thresholds


def in_force_edge_thresholds(
    artifacts_dir: Path | str, ledger_dir: Path | str
) -> EdgeThresholds | None:
    """The in-force generation's edge-tier thresholds for the website's bands (D-36).

    Args:
        artifacts_dir: The artifacts directory whose ``latest.json`` the cache build serves --
            the SAME directory the caller hands to ``api.cache.populate_cache``.
        ledger_dir: The forward ledger directory holding ``refit/`` (``LEDGER_DIR`` in
            production).

    Returns:
        None when *artifacts_dir* holds no ``latest.json`` or its ids are generation 0 (the
        row-19 constants then band, exactly as before); otherwise the generation's bet-rule
        record thresholds, ``{target -> (high, medium) | None}`` for every canonical target.

    Raises:
        UnknownGenerationError: ``latest.json`` names ids no generation produced.
        RefitRecordFormatError: the generation records, or its bet-rule file, cannot be trusted.
    """
    artifacts = Path(artifacts_dir)
    if not (artifacts / "latest.json").exists():
        return None
    return _generation_edge_thresholds(
        in_force_generation(resolve_production_artifacts(artifacts), ledger_dir)
    )


def decided_edge_thresholds(
    entries: Sequence[LedgerEntry], ledger_dir: Path | str
) -> dict[str, EdgeThresholds | None]:
    """Each live-decided game's edge-band thresholds: those of the generation that decided it.

    Record section 12 (D-36): "a game's displayed band is the band its live bet rule used". The
    generation is derived from each live row's stamped model id (:func:`generation_of_row`). The
    three targets of a game are decided in one run; a game decided again under a later week's row
    takes its LAST decision in file order. Shadow rows are never read: they decide no bet.

    Args:
        entries: The verified ledger, in file order (``forward_ledger.store.read_entries``).
        ledger_dir: The ledger directory holding ``refit/``. Read only when a live row names a
            re-fit generation's id, so a ledger of generation-0 rows reads no re-fit record.

    Returns:
        ``game_id -> thresholds``: None for a game decided by generation 0, or by the pre-row-19
        models of weeks 3-4 (both band with the row-19 constants), else the deciding generation's
        bet-rule thresholds. A game with no live row is absent.

    Raises:
        UnknownGenerationError: a live row's stamped id belongs to no generation.
        RefitRecordFormatError: the generation records, or a bet-rule file, cannot be trusted.
    """
    live_rows = [
        entry.immutable
        for entry in entries
        if entry.kind == ENTRY_KIND_ROW and entry.immutable["arm"] == ARM_LIVE
    ]
    pinned = _model_ids(generation_zero_ids())
    names_a_refit = any(
        row["model_artifact_id"] is not None and row["model_artifact_id"] not in pinned
        for row in live_rows
    )
    generations = (
        installed_generations(ledger_dir, skip_torn_tail=True) if names_a_refit else ()
    )
    by_number = {generation.number: generation for generation in generations}
    thresholds_by_number: dict[int, EdgeThresholds | None] = {}
    decided: dict[str, EdgeThresholds | None] = {}
    for row in live_rows:
        number = generation_of_row(row["model_artifact_id"], ARM_LIVE, generations)
        if number is None or number == GENERATION_ZERO:
            decided[str(row["game_id"])] = None
            continue
        if number not in thresholds_by_number:
            thresholds_by_number[number] = _generation_edge_thresholds(
                by_number[number]
            )
        decided[str(row["game_id"])] = thresholds_by_number[number]
    return decided


def _model_ids(resolved: ResolvedArtifacts) -> frozenset[str]:
    """The three model ids of *resolved* (the blend and converter name no model)."""
    return frozenset(resolved.model_id_for(target) for target in _MODEL_TARGETS)


def generation_of_row(
    model_artifact_id: str | None,
    row_arm: str,
    generations: Sequence[Generation] = (),
) -> int | None:
    """The generation whose models produced a row, from the row's stamped model id.

    Args:
        model_artifact_id: The row's stamped ``model_artifact_id`` (None for weeks 3-4).
        row_arm: The row's ``arm`` (``live`` or ``shadow``).
        generations: The installed generations (:func:`installed_generations`); generation 0 is
            always the pin, whether or not it is in the sequence.

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
    installs = [g for g in generations if g.number != GENERATION_ZERO]
    for generation in installs:
        if model_artifact_id in _model_ids(generation.ids):
            return generation.number
    msg = (
        f"a {row_arm} row is stamped with model id {model_artifact_id!r}, which is neither one of "
        f"the frozen pin's ids {sorted(pinned)} nor an id of any installed generation "
        f"{[generation.number for generation in installs]}."
    )
    raise UnknownGenerationError(msg)
