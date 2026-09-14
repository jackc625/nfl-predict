"""Permanent doc-drift guard for the repo-root HISTORICAL-WEATHER-READOUT.md (33.1-SPEC.md R8).

A PERMANENT COMMITTED TEST, not a throwaway ``scripts/check_*.py``. It mirrors
``tests/unit/test_signal_lift_readout_md.py``, which is this repository's committed doc-drift-guard
pattern, and guards the R8 deliverable against five failure modes:

  - the file is missing or not at the repository root,
  - the content carries non-ASCII characters (CLAUDE.md hard constraint),
  - a required section was silently dropped -- the markers are SUBSTRINGS, so a benign reword does
    not trip the guard but a dropped section does,
  - a forbidden phrase appears: the three suite-claim phrasings from
    ``tests.phase33_state.FORBIDDEN_SUITE_CLAIM_PHRASES``, or any accuracy or profitability claim
    (33.1-SPEC.md prohibition 4),
  - a required phrase was removed -- the readout stops saying that no model was re-fit, stops saying
    ``artifacts/latest.json`` is byte-identical, stops correcting the 534-games record, stops naming
    an instruction in each of ``33-14-PLAN.md`` and ``33-15-PLAN.md``, or stops carrying one of the
    disclosures the phase is obliged to carry.

WHAT IT DELIBERATELY DOES NOT DO, AND WHY (the D29-06-02 lesson, applied at authoring time rather
than after). It does NOT re-run a harness and does NOT pin a point estimate to live gold. Every
numeric claim is asserted against the DOCUMENT and against the committed ``tests/phase33_state``
slots the document quotes, which are records rather than moving measurements. Plan 33.1-08 had to
generation-gate four existing guards for exactly this reason; this one is written so it never needs
gating, and an AST check in the plan's verification asserts it imports no harness module.

THE ONE FILE IT READS OUTSIDE THE DOCUMENT is ``artifacts/latest.json``, and only to take a sha256 of
its bytes. That is the phase's byte-identity fence, which 33.1-11-PLAN.md's prohibitions table names
in this module by name. It reads nothing under ``data/`` and it never writes.

TWO CONSISTENCY TESTS rather than substring checks, because two of the readout's claims can each be
individually well-formed and jointly wrong: the readout's R7 claim must agree with
``WEATHER_BRIDGE_R7_DISPOSITION["status"]``, and the deployed manifest must still digest to the value
recorded at this phase's start.

ASCII only, no emoji (CLAUDE.md hard constraint).
"""

from __future__ import annotations

import hashlib
from pathlib import Path

import pytest

from tests import phase33_state

REPO_ROOT = Path(__file__).resolve().parents[2]
READOUT_MD = REPO_ROOT / "HISTORICAL-WEATHER-READOUT.md"
LIVE_MANIFEST = REPO_ROOT / "artifacts" / "latest.json"

# Required section markers. Substrings of each section heading, so a reword inside a section does
# not trip the guard but dropping the section does.
REQUIRED_SECTION_MARKERS: tuple[str, ...] = (
    "READ THIS FIRST -- the Wave-15 instruction that is now wrong",
    "The measured weather rung",
    "The cross-check against its pre-registered expectation",
    "The re-derived training window rule",
    "The Phase 33 instructions the corrected inputs have invalidated",
    "no model was re-fit, and nothing has been promoted",
    "the 2026 gold-weather bridge is BOUNDED",
    "The disclosures this phase is obliged to carry",
    "What Phase 33 should do next",
)

# Required phrases. Each is a claim the readout is OBLIGED to keep making; removing one is the
# failure mode this list exists to catch.
REQUIRED_PHRASES: tuple[str, ...] = (
    # The two plain statements R8 demands.
    "no model was re-fit",
    "byte-identical",
    # The two Phase-33 plans whose instructions it invalidates.
    "33-14-PLAN.md",
    "33-15-PLAN.md",
    "same feature set",
    # The corrected record, not quietly dropped.
    "actually describe is the FEATURE-SELECTION and HYPERPARAMETER window",
    # The disclosures.
    "in-sample",
    "never promoted",
    "Ninety gold columns",
    # The probe's explicit n, pinned. A committed record rather than a moving measurement: the
    # population cannot grow, because it is every row in this repository whose provenance is known
    # or claimed to be the forecast feed. Reporting the probe WITHOUT its sample size is how a
    # directional probe on 28 games turns into an estimate somebody acts on.
    "Sample size: n = 28 at most",
    "D33.1-R1",
    "D33.1-R2",
    "D33.1-R3",
    "frozen record",
    "wp_20260824_113325",
    "does not fix",
    "Wave 15",
    # Attribution is a licence requirement, not a courtesy.
    "CC BY 4.0",
)

# The accuracy and profitability claims 33.1-SPEC.md prohibition 4 forbids. The three suite-claim
# phrasings are NOT spelled here -- they come from tests.phase33_state, which keeps this module
# honest under the same rule it enforces rather than exempt from it.
FORBIDDEN_PERFORMANCE_PHRASES: tuple[str, ...] = (
    "improves accuracy",
    "more profitable",
    "more accurate",
    "profitability improvement",
    "was promoted",
    "was deployed",
)


def forbidden_phrases() -> tuple[str, ...]:
    """Every phrase the readout may not contain, suite claims included."""
    return (
        tuple(phase33_state.FORBIDDEN_SUITE_CLAIM_PHRASES)
        + FORBIDDEN_PERFORMANCE_PHRASES
    )


def readout_problems(path: Path) -> list[str]:
    """Every doc-drift problem in the readout at *path*, as actionable strings.

    Returns problems rather than raising, so one run reports every failure mode at once instead of
    stopping at the first -- a guard that names one missing section per run is a guard somebody
    fixes three times.

    Each problem string CONTAINS the marker, phrase or word it is about, so the assertion a later
    reader sees tells them what to put back rather than only that something is wrong.

    Args:
        path: The readout to check. The committed one at the repository root in normal use, and a
            temporary copy in ``tmp_path`` for every planted-violation control.

    Returns:
        A list of actionable problem descriptions; empty when the document is healthy.
    """
    if not path.is_file():
        return [
            f"missing: {path}. The R8 deliverable must exist at the repository root as "
            f"{READOUT_MD.name}, beside GATED-REFIT-READOUT.md and its siblings -- NOT under the "
            "gitignored .planning/ directory."
        ]

    content = path.read_text(encoding="utf-8")
    problems: list[str] = []

    if not content.isascii():
        offenders = sorted(
            {character for character in content if not character.isascii()}
        )
        problems.append(
            f"non-ascii characters in {path.name}: {offenders}. CLAUDE.md requires ASCII only; "
            "arrows are '->', dashes are '--' and quotes are straight."
        )

    problems.extend(
        f"required section marker dropped: {marker!r}"
        for marker in REQUIRED_SECTION_MARKERS
        if marker not in content
    )

    problems.extend(
        f"required phrase removed: {phrase!r}"
        for phrase in REQUIRED_PHRASES
        if phrase not in content
    )

    lowered = content.lower()
    problems.extend(
        f"forbidden phrase present: {phrase!r}. This phase re-fit no model and moved no "
        "production pointer, so it claims no accuracy or profitability change (33.1-SPEC.md "
        "prohibition 4), and no file in this phase may claim the suite reaches a clean summary "
        "line while five tests are deliberately red."
        for phrase in forbidden_phrases()
        if phrase.lower() in lowered
    )

    return problems


def _plant(tmp_path: Path, *, mutate) -> Path:
    """A TEMPORARY copy of the committed readout with *mutate* applied. Never the real file."""
    copy = tmp_path / READOUT_MD.name
    copy.write_text(mutate(READOUT_MD.read_text(encoding="utf-8")), encoding="utf-8")
    return copy


class TestTheReadoutIsHealthy:
    """The committed document passes every check."""

    def test_the_file_exists_at_the_repository_root(self) -> None:
        """The R8 deliverable is at the repo root, NOT under the gitignored .planning/."""
        assert READOUT_MD.is_file(), f"missing: {READOUT_MD}"

    def test_the_committed_readout_has_no_problems(self) -> None:
        """No missing section, no missing phrase, no forbidden phrase, no non-ASCII."""
        assert readout_problems(READOUT_MD) == []

    def test_the_content_is_ascii(self) -> None:
        """ASCII only (CLAUDE.md hard constraint)."""
        assert READOUT_MD.read_text(encoding="utf-8").isascii()


class TestTheReadoutMakesNoPerformanceClaim:
    """33.1-SPEC.md prohibition 4: this phase re-fit no model, so it claims no accuracy gain."""

    def test_no_forbidden_phrase_appears(self) -> None:
        """Neither a suite claim nor an accuracy or profitability claim."""
        lowered = READOUT_MD.read_text(encoding="utf-8").lower()
        present = [p for p in forbidden_phrases() if p.lower() in lowered]
        assert not present, (
            f"HISTORICAL-WEATHER-READOUT.md carries forbidden phrases: {present}"
        )


class TestTheDeployedManifestIsByteIdentical:
    """The fence the readout's central claim rests on, checked rather than quoted."""

    def test_latest_json_digests_to_the_value_recorded_at_phase_start(self) -> None:
        """``artifacts/latest.json`` still digests to the value recorded before this phase ran."""
        recorded = phase33_state.FINAL_FIT_NOT_RUN_IN_PHASE_331[
            "latest_json_digest_before"
        ]
        measured = hashlib.sha256(LIVE_MANIFEST.read_bytes()).hexdigest()
        assert measured == recorded, (
            "artifacts/latest.json has MOVED. The readout states plainly that no model was re-fit "
            "and that nothing reached production; that statement rests on this digest. Either the "
            "readout is now false or a production swap happened outside the record."
        )


class TestTheR7ClaimAgreesWithTheRecord:
    """A CONSISTENCY test, not a substring check.

    A readout saying the bridge is bounded while ``WEATHER_BRIDGE_R7_DISPOSITION`` records the
    opposite is the blurring Ruling V2 exists to prevent, and neither half's own phrase check would
    catch it: both halves are individually well-formed and jointly false.
    """

    def test_the_readout_claims_exactly_what_the_disposition_records(self) -> None:
        """Bounded in the document if and only if satisfied in the record."""
        lowered = READOUT_MD.read_text(encoding="utf-8").lower()
        status = phase33_state.WEATHER_BRIDGE_R7_DISPOSITION["status"]
        claims_bounded = "bounded" in lowered and "unmet" not in lowered
        assert claims_bounded == (status == "satisfied"), (
            f"the readout's R7 claim disagrees with WEATHER_BRIDGE_R7_DISPOSITION['status']="
            f"{status!r}: claims_bounded={claims_bounded}"
        )


class TestThePlantedViolationsFire:
    """Five planted controls, one per failure mode. Without them this guard asserts nothing.

    Every control runs against a TEMPORARY copy in ``tmp_path`` and never against the committed
    document. A guard that has only ever seen a healthy document is indistinguishable from a guard
    that is not wired up.
    """

    def test_planted_missing_file_is_flagged_by_name(self, tmp_path: Path) -> None:
        """Pointing the guard at an absent path fails, naming the expected repo-root location."""
        problems = readout_problems(tmp_path / "HISTORICAL-WEATHER-READOUT.md")
        assert problems, "an absent readout was not flagged"
        assert any("missing" in problem.lower() for problem in problems)
        assert any("HISTORICAL-WEATHER-READOUT.md" in problem for problem in problems)

    @pytest.mark.parametrize(
        "marker",
        (
            "The measured weather rung",
            "The Phase 33 instructions the corrected inputs have invalidated",
            "the 2026 gold-weather bridge is BOUNDED",
        ),
    )
    def test_planted_dropped_section_is_flagged_by_name(
        self, tmp_path: Path, marker: str
    ) -> None:
        """Removing any one required section marker fails, naming that marker."""
        planted = _plant(tmp_path, mutate=lambda text: text.replace(marker, "REMOVED"))
        problems = readout_problems(planted)
        assert problems, f"a dropped section {marker!r} was not flagged"
        assert any(marker in problem for problem in problems)

    def test_planted_non_ascii_is_flagged(self, tmp_path: Path) -> None:
        """Inserting a non-ASCII character fails."""
        planted = _plant(tmp_path, mutate=lambda text: text + "\n\nnon-ascii: —\n")
        problems = readout_problems(planted)
        assert problems, "a non-ASCII character was not flagged"
        assert any("ascii" in problem.lower() for problem in problems)

    def test_planted_forbidden_phrase_is_flagged_by_name(self, tmp_path: Path) -> None:
        """Inserting a forbidden phrase fails, naming the phrase."""
        phrase = FORBIDDEN_PERFORMANCE_PHRASES[0]
        planted = _plant(
            tmp_path, mutate=lambda text: f"{text}\n\nThis rebuild {phrase}.\n"
        )
        problems = readout_problems(planted)
        assert problems, "a forbidden phrase was not flagged"
        assert any(phrase in problem for problem in problems)

    def test_planted_forbidden_suite_claim_is_flagged(self, tmp_path: Path) -> None:
        """A suite-claim phrasing sourced from the manifest is flagged too, never hard-coded here."""
        phrase = phase33_state.FORBIDDEN_SUITE_CLAIM_PHRASES[0]
        planted = _plant(
            tmp_path, mutate=lambda text: f"{text}\n\nThe suite reports {phrase}.\n"
        )
        problems = readout_problems(planted)
        assert problems, "a suite-claim phrasing was not flagged"
        assert any(phrase in problem for problem in problems)

    def test_planted_removed_required_phrase_is_flagged_by_name(
        self, tmp_path: Path
    ) -> None:
        """Removing the 'no model was re-fit' statement fails, naming that phrase."""
        planted = _plant(
            tmp_path,
            mutate=lambda text: text.replace(
                "no model was re-fit", "the phase finished"
            ),
        )
        problems = readout_problems(planted)
        assert problems, "a removed required phrase was not flagged"
        assert any("no model was re-fit" in problem for problem in problems)


class TestTheGuardIsNotVacuous:
    """The lists it checks are non-empty, so a green run means something was checked."""

    def test_the_marker_and_phrase_lists_are_populated(self) -> None:
        """An empty list would make every section and phrase check pass for free."""
        assert len(REQUIRED_SECTION_MARKERS) >= 9
        assert len(REQUIRED_PHRASES) >= 18
        assert len(forbidden_phrases()) >= 9
