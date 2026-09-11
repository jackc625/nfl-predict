"""The FROZEN upstream-revision verdict vocabulary (D32-12, Plan 32-03).

WHAT THIS VOCABULARY IS FOR
---------------------------
Every verdict this project records about an upstream revision stamps TWO things: an EVENT
CLASS naming WHAT happened (a sealed season moved; a live capture moved; a live capture
moved on a week that was ALREADY GRADED; the detector could not see) and a SEVERITY naming
HOW LOUD it is. They are separate on purpose. "What happened" is a fact about the data and
never changes; "how loud" is a calibration this project can retune without rewriting the
history of what it observed.

WHO IMPORTS IT
--------------
* Phase 32 -- the sealed probe and the live-capture detector, which stamp the verdicts.
* Phase 34 -- the forward bet ledger's correction blocks, which key off
  ``LIVE_REVISION_GRADED`` and the ``CORRECTION_OWED`` data below.
* Phase 35 -- notification routing, which decides what reaches a human by SEVERITY rather
  than by re-deriving its own words from the event class.

WHY IT LIVES HERE AND NOT IN THE CLI OR IN ``data/upstream_pin.py``
--------------------------------------------------------------------
``32-RESEARCH.md``'s Architectural Responsibility Map: a consumer must not have to drag a
dependency it does not want in order to read a word. Putting the vocabulary in
``scripts/capture_live_season.py`` would make Phase 35's notifier import a CLI; putting it in
``data/upstream_pin.py`` would make it import the whole pin machinery, nflreadpy deferral and
all. This module imports ``__future__`` and ``enum``, performs no I/O, and holds no behaviour
beyond one pure rank lookup, so importing it costs nothing and can never have a side effect.

REVERSIBILITY -- READ THIS BEFORE RENAMING ANYTHING
-----------------------------------------------------
This file is a contract, not a convenience. Renaming an event class or a severity later
touches every consumer AND every already-written verdict: the values below are serialised
verbatim into ``config/upstream_live/<season>.json`` capture entries and into the committed
sealed probe log, which accumulate all season. A rename makes a season's recorded history
stop being comparable end to end, and there is no migration that can honestly repair it --
the old entries were computed under the old meaning. ``32-CONTEXT.md`` rates D32-12 COSTLY
for exactly this reason. Add a member if a genuinely new thing can happen; do not rename one.

ASCII only, no emoji (CLAUDE.md hard constraint).
"""

from __future__ import annotations

from enum import StrEnum

__all__ = [
    "CORRECTION_OWED",
    "CORRECTION_OWED_SCOPE",
    "DEFAULT_SEVERITY",
    "SEVERITY_ORDER",
    "VERDICT_SCHEMA_VERSION",
    "RevisionEventClass",
    "RevisionSeverity",
    "severity_rank",
]


# Both enums are ``StrEnum`` and not the plain ``Enum`` of ``utils/alert_manager.py``'s
# ``AlertLevel``. That is the only difference from it, and the reason is mechanical: a
# StrEnum member IS a str, so ``json.dumps({"severity": RevisionSeverity.CRITICAL})``
# writes ``"critical"`` with no custom encoder and no ``.value`` at the write site. These
# values land in committed JSON manifests and in the JSONL probe log; a ``.value`` forgotten
# at ONE write site would put an unreadable ``"RevisionSeverity.CRITICAL"`` into a committed
# record. ``data/schemas.py::VenueRoof`` is the same choice made for the same reason.


class RevisionEventClass(StrEnum):
    """WHAT happened, as one machine-readable word. Seven members, closed (D32-12)."""

    # A SEALED (2001-2025) season's upstream bytes no longer match the pin. Emitted by the
    # Phase-32 sealed probe; consumed by the committed sealed log, by Phase 35's routing,
    # and by any later re-capture decision. Never blocking: the pipeline reads pinned bytes,
    # not upstream, so the run is provably unaffected (D32-09).
    SEALED_REVISION = "sealed_revision"

    # A live (2026+) capture moved on a week whose bets were ALREADY GRADED. Emitted by the
    # live detector when ``data.graded_weeks.graded_weeks`` reports the moved week as
    # settled; consumed by Phase 34, which appends a ledger correction block. ARCHITECTURE
    # 5.4 forbids re-grading, so this class is the ONLY route by which a post-grading data
    # correction is ever acknowledged.
    LIVE_REVISION_GRADED = "live_revision_graded"

    # A live capture moved on a week that is not yet graded. Emitted by the live detector on
    # the ordinary case; consumed by the capture entry's own verdict block.
    #
    # B3 CALIBRATION, STATED EXPLICITLY: an in-season revision of the CURRENT week is
    # NORMAL. nflverse restates the running week's play-by-play routinely, so this event
    # fires most weeks of most seasons. It therefore must NOT share a severity with the
    # graded case or with a sealed move -- a detector that reports CRITICAL every week is
    # wallpaper by week three, and the first genuinely important finding arrives into an
    # inbox nobody reads any more.
    LIVE_REVISION = "live_revision"

    # There is no earlier capture of this (season, week) to diff against, so nothing can be
    # said about movement. Emitted on the first capture of a week; consumed by the capture
    # entry. It is deliberately NOT ``CLEAN``: "nothing to compare" and "compared and found
    # no change" are different claims and a season of entries must be able to tell them
    # apart after the fact.
    NO_PRIOR_CAPTURE = "no_prior_capture"

    # The observed upstream signature matches a divergence the owner has already ruled on
    # and acknowledged (D32-10). Emitted by the sealed probe when the recorded
    # ``acknowledgement`` signature still matches; consumed by the sealed log and by
    # Phase 35, which routes it nowhere. A NEW move on top of an acknowledgement is a
    # ``SEALED_REVISION`` again, not this.
    KNOWN_DIVERGENCE_STABLE = "known_divergence_stable"

    # The detector looked and found no movement. Emitted by both detectors; consumed by the
    # committed logs, where an unbroken run of CLEAN entries is itself the proof the
    # detector was alive all season (D32-08).
    CLEAN = "clean"

    # The detector could not tell -- a probe network error, an unreadable store, a missing
    # dependency. Emitted wherever an honest answer is unavailable; consumed by every
    # reader. PITFALLS F2: a dead detector and a healthy system must never be the same
    # observable, so this is a recorded verdict carrying its reason and NEVER a silent skip
    # and NEVER ``CLEAN``.
    UNKNOWN = "unknown"


class RevisionSeverity(StrEnum):
    """HOW LOUD a verdict is. Three members, ordered by ``SEVERITY_ORDER`` below."""

    # Recorded, and that is all. The routine case.
    INFORMATIONAL = "informational"

    # Worth a human's attention, but not a confirmed finding about the data.
    WARNING = "warning"

    # A confirmed move in bytes this project's published numbers depend on. Loud, recorded,
    # and still NEVER blocking (D32-09): blocking a run that is demonstrably correct is how
    # a detector earns an override flag and then gets ignored.
    CRITICAL = "critical"


# The frozen severity ladder, ascending. ``severity_rank`` is its index and nothing else.
SEVERITY_ORDER: tuple[RevisionSeverity, ...] = (
    RevisionSeverity.INFORMATIONAL,
    RevisionSeverity.WARNING,
    RevisionSeverity.CRITICAL,
)


# The severity each event class carries unless a caller deliberately overrides it.
#
# ``UNKNOWN -> WARNING`` IS DELIBERATE, AND DELIBERATELY NOT ``CRITICAL``. The honest claim
# an UNKNOWN makes is "I could not tell", which is strictly louder than clean -- PITFALLS F2
# -- and strictly quieter than a confirmed sealed move, because no movement has actually
# been observed. D32-09 makes neither of them blocking anyway, so overstating UNKNOWN buys
# no safety and costs the credibility of the CRITICAL beside it.
#
# ``KNOWN_DIVERGENCE_STABLE -> INFORMATIONAL`` is the whole point of D32-10: an acknowledged,
# stable divergence must not re-fire at the volume of a new one.
#
# The key set is asserted equal to ``set(RevisionEventClass)`` by
# ``tests/unit/test_revision_events.py``, so a member added without a severity FAILS the
# suite rather than reaching a ``KeyError`` inside a detector on a Friday evening.
DEFAULT_SEVERITY: dict[RevisionEventClass, RevisionSeverity] = {
    RevisionEventClass.SEALED_REVISION: RevisionSeverity.CRITICAL,
    RevisionEventClass.LIVE_REVISION_GRADED: RevisionSeverity.CRITICAL,
    RevisionEventClass.LIVE_REVISION: RevisionSeverity.INFORMATIONAL,
    RevisionEventClass.NO_PRIOR_CAPTURE: RevisionSeverity.INFORMATIONAL,
    RevisionEventClass.KNOWN_DIVERGENCE_STABLE: RevisionSeverity.INFORMATIONAL,
    RevisionEventClass.CLEAN: RevisionSeverity.INFORMATIONAL,
    RevisionEventClass.UNKNOWN: RevisionSeverity.WARNING,
}


# Stamped on every recorded verdict so a reader years later can tell WHICH rule computed it.
# Bump only when the verdict's SHAPE changes; the severity table above can be retuned within
# a version because it is a calibration, not a schema.
VERDICT_SCHEMA_VERSION: int = 1


# The two queryable keys D32-12 names. Phase 32 only RAISES the obligation, as DATA on the
# verdict -- ``{CORRECTION_OWED: True, CORRECTION_OWED_SCOPE: {...the graded weeks...}}`` --
# and builds nothing. Phase 34 DISCHARGES it by appending a correction block to the forward
# ledger, because ARCHITECTURE 5.4 rules that a corrected score arriving after a bet was
# graded can never re-grade it. Recording the obligation as prose in a log line would make
# it unfindable; recording it as a key makes "which corrections are still owed" a query.
CORRECTION_OWED: str = "correction_owed"
CORRECTION_OWED_SCOPE: str = "correction_owed_scope"


def severity_rank(severity: RevisionSeverity) -> int:
    """Return *severity*'s index in :data:`SEVERITY_ORDER` -- higher means louder.

    The module's ONLY function, and pure. It exists so that "the graded case escalates" is a
    number a test can check rather than a sentence in a docstring.

    The ``isinstance`` guard is load-bearing and not defensive boilerplate.
    :class:`RevisionSeverity` is a ``StrEnum``, so the bare string ``"critical"`` compares
    EQUAL to ``RevisionSeverity.CRITICAL`` and a naive ``SEVERITY_ORDER.index(severity)``
    would happily rank it. Accepting bare strings would let a typo'd word enter a comparison
    silently, which is exactly what a frozen vocabulary exists to prevent -- the same refusal
    ``backtest/group_gate.py``'s ``_KNOWN_VERDICTS`` guard makes, naming the vocabulary in
    the message so the reader learns the permitted set from the failure.

    Args:
        severity: A :class:`RevisionSeverity` member. Anything else is refused.

    Returns:
        ``0`` for INFORMATIONAL, ``1`` for WARNING, ``2`` for CRITICAL.

    Raises:
        ValueError: If *severity* is not a :class:`RevisionSeverity` member.
    """
    if not isinstance(severity, RevisionSeverity):
        permitted = ", ".join(member.name for member in SEVERITY_ORDER)
        msg = (
            f"{severity!r} is not a RevisionSeverity member. severity_rank ranks only the "
            f"frozen SEVERITY_ORDER vocabulary ({permitted}); a bare string is refused "
            "because RevisionSeverity is a StrEnum and would compare equal to a member, "
            "letting a misspelled word rank silently."
        )
        raise ValueError(msg)
    return SEVERITY_ORDER.index(severity)
