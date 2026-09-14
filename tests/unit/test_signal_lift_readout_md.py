"""Permanent doc-drift guard for the repo-root SIGNAL-LIFT-READOUT.md (Phase 28, SIG-05 / D-20).

Mirrors ``tests/unit/test_ou_divergence_diagnosis_md.py`` (the committed doc-drift-guard pattern):
a PERMANENT committed test, NOT a throwaway ``scripts/check_*.py``. It guards the SIG-05 / D-20
deliverable ``SIGNAL-LIFT-READOUT.md`` against four failure modes:

  - the file is missing or not at the repo root,
  - the content carries non-ASCII (emoji / cp1252-hostile) characters (CLAUDE.md hard constraint),
  - a required section (the per-group grids, the keep/drop summary, the situational caveat, the
    burned-holdout caveat, the screen-not-deploy framing) was silently dropped,
  - the SCREEN-NOT-DEPLOY invariant was violated -- the doc must say "carried to Phase 30" and must
    NEVER say "deployed" / "proven" (a screen is not a deploy decision, D-01 / D-20).

The deeper anti-rot guard (``TestReadoutMatchesHarness``) RUNS ``run_signal_lift_screen`` and
asserts the RULING the doc records as CURRENT is the ruling the committed harness returns, so the
doc cannot silently drift from the harness (the Phase-26 doc-drift convention). As of 2026-09-05
that current ruling is DROP, not the 2026-06-29 KEEP: see ``_CURRENT_*`` below and Section 0b of
the readout.

ASCII only, no emoji (CLAUDE.md).
"""

from __future__ import annotations

from pathlib import Path

# Repo root resolved from this file: tests/unit/test_signal_lift_readout_md.py -> repo root.
REPO_ROOT = Path(__file__).resolve().parents[2]
READOUT_MD = REPO_ROOT / "SIGNAL-LIFT-READOUT.md"

# Gold presence skip-guard: the harness-run validation needs the Plan 28-06 widened gold.
_GOLD_OU_PATH = REPO_ROOT / "data" / "gold" / "features_ou.parquet"
_ODDS_PATH = REPO_ROOT / "data" / "silver" / "odds_snapshot.parquet"

# Required section markers (D-20 required-content list). A substring of each section so a benign
# reword does not trip the guard but a dropped section does.
_REQUIRED_SECTION_MARKERS = (
    "METHOD",  # 0 -- the in-process walk-forward correction
    "Per-target incremental-CLV lift grid",  # 1
    "Injury group",  # per-group grid
    "Snap group",
    "Situational group",
    "Keep/drop summary",  # 2
    "Multiplicity note",  # the 3x3 grid note
    "Situational caveat",  # 3 (SC3 / D-17)
    "Burned-holdout caveat",  # 4 (D-18e)
    "Screen-not-deploy framing",  # 5 (D-01 / D-20)
)

# The screen-not-deploy invariant: the carry phrase MUST be present; the over-claim words MUST NOT.
_REQUIRED_PHRASES = (
    "carried to Phase 30",
    "priced-in",
)
_FORBIDDEN_WORDS = ("deployed", "proven")

# ---------------------------------------------------------------------------
# The RECONCILED state (2026-09-05), which this guard now pins.
#
# The Plan 31-11 full gold rebuild corrected 68 protected-window games that had been graded
# against a FABRICATED 0.0 market line (the LAR -> LA odds-key orphan; register DEF-31-09), which
# moved the ATS and O/U LABELS those games carry. On that corrected gold the Phase-28 screen no
# longer returns its recorded KEEP: situational carries a D-05 veto on OU, and so do injury (WP)
# and snap (ATS, OU). The readout was reconciled -- Section 0b records the flip with its cause,
# the 2026-06-29 anchor is left standing as the historical record -- and this guard moved with it.
#
# It was NOT weakened to do so. The invariant is unchanged in kind: the doc's CURRENT ruling must
# be the harness's ruling. Only the ruling being pinned changed, because the ruling changed. A
# return to KEEP now fails just as loudly as the flip to DROP did, and for the same reason -- it
# would mean the doc and the harness disagree again.
# ---------------------------------------------------------------------------

# The 2026-06-29 Phase-28 anchor, which the doc must keep recording rather than overwrite.
_HISTORICAL_ANCHOR = "+0.177334"

# The dated 2026-09-05 measurement the doc must keep recording, and the sentence that states
# which reading of the document is current. Asserted against the DOC, never re-asserted against
# a re-run of the harness -- pinning a point estimate to moving gold is the mistake this guard
# already made once (D29-06-02).
_CURRENT_MEASURED_DELTA = "-0.3203552582994336"
_CURRENT_RULING_MARKER = (
    "The CURRENT ruling of this screen, on 2026-09-05 corrected gold, is **DROP for all "
    "three groups**"
)
_CURRENT_CAUSE_MARKER = "FABRICATED 0.0 market line"


def _read_readout() -> str:
    """Read SIGNAL-LIFT-READOUT.md from the repo root."""
    return READOUT_MD.read_text(encoding="utf-8")


class TestReadoutExists:
    """SIGNAL-LIFT-READOUT.md must exist at the repo root and be ASCII-only."""

    def test_file_exists_at_repo_root(self) -> None:
        """The SIG-05 / D-20 deliverable is present at the repo root (NOT under .planning/)."""
        assert READOUT_MD.is_file(), f"missing: {READOUT_MD}"

    def test_content_is_ascii(self) -> None:
        """Content is ASCII-only (no emoji / non-ASCII, per CLAUDE.md)."""
        content = _read_readout()
        assert content.isascii(), "SIGNAL-LIFT-READOUT.md contains non-ASCII characters"


class TestReadoutSections:
    """All required sections + the per-group grids must be present."""

    def test_all_required_section_markers_present(self) -> None:
        """Every required section marker is present; names any missing so a drop is actionable."""
        content = _read_readout()
        missing = [m for m in _REQUIRED_SECTION_MARKERS if m not in content]
        assert not missing, (
            f"SIGNAL-LIFT-READOUT.md missing required sections: {missing}"
        )

    def test_per_target_grid_covers_all_three_targets(self) -> None:
        """Each group's grid reports all three targets (WP / ATS / OU)."""
        content = _read_readout()
        for target_label in ("| WP", "| ATS", "| OU"):
            assert target_label in content, (
                f"the per-target grid is missing a {target_label!r} row"
            )

    def test_covered_spans_annotated(self) -> None:
        """Each group's covered span is annotated beside its grid (D-18d)."""
        content = _read_readout()
        for span in ("injuries 2009+", "snaps 2013+"):
            assert span in content, f"missing covered-span annotation {span!r}"


class TestScreenNotDeployInvariant:
    """The doc must say 'carried to Phase 30' and must NEVER say 'deployed' / 'proven' (D-01/D-20)."""

    def test_required_carry_phrases_present(self) -> None:
        """The screen-not-deploy carry phrase + the situational priced-in caveat are present."""
        content = _read_readout()
        missing = [p for p in _REQUIRED_PHRASES if p not in content]
        assert not missing, (
            f"SIGNAL-LIFT-READOUT.md missing required phrases: {missing}"
        )

    def test_no_over_claim_words(self) -> None:
        """The over-claim words 'deployed' / 'proven' must NOT appear (a screen is not a deploy)."""
        lowered = _read_readout().lower()
        present = [w for w in _FORBIDDEN_WORDS if w in lowered]
        assert not present, (
            f"SIGNAL-LIFT-READOUT.md over-claims a screen as a deploy decision: {present} "
            "(D-01 / D-20 -- use 'carried to Phase 30', never 'deployed' / 'proven')"
        )

    def test_burned_holdout_caveat_recorded(self) -> None:
        """The D-18e burned-holdout caveat is recorded (noted, not a blocker)."""
        content = _read_readout()
        assert "D26-09" in content or "burned" in content.lower(), (
            "the D-18e burned-holdout caveat is missing"
        )


# The harness-reproduction class that stood here was DELETED on 2026-09-12 by owner
# instruction. It re-ran the analysis harness against live gold and asserted the
# committed point estimates still reproduced. Two defects made it undefendable:
#   1. NOT DETERMINISTIC. The situational-OU delta measured 0.0074 / 0.4424 / 0.4784
#      at 4 / 1 / 8 BLAS threads and only reproduced the committed value at 12. It
#      cannot detect drift because it drifts on its own; the quantity is a difference
#      between noisy per-season estimates, so floating-point reduction order moves it
#      further than the signal does.
#   2. Plan 33.1-08 had already concluded the same thing and designed the successor:
#      generation-gate the harness half so it SKIPS when gold moves. See that plan and
#      D29-06-02 -- 'pinning a point estimate to moving gold is the mistake this guard
#      class made once'.
# The DOC-level assertions in this file are untouched and still run unconditionally.

# ---------------------------------------------------------------------------
# THE GENERATION SEAM, AND WHY THIS MODULE DOES NOT CALL IT -- Plan 33.1-08
# Task 2, 2026-09-14.
#
# Plan 33.1-08 was written to GENERATION-GATE the harness-reproduction class
# above rather than let a gold rebuild turn it red. By the time the plan ran the
# class was already gone, deleted on 2026-09-12 for a different and better
# reason: it was not deterministic, so it could not detect drift because it
# drifted on its own. A seam that refuses to compare across gold generations
# cannot help a measurement that disagrees with itself at a fixed generation.
#
# The seam was still built -- tests/gold_generation.py -- because two OTHER
# harness reproductions did redden on the rung-3 rebuild and needed it, and
# because Phase 33's Wave 14 Elo rung will hit the same wall again. It is
# recorded here so a reader of this file can find it:
#
#   tests.gold_generation.require_gold_generation(expected_key, *, reading,
#       moved_by, recorded_in) -- skips, never fails, and names all three in the
#       skip message.
#   tests.phase33_state.GOLD_GENERATION_AFTER_WEATHER_RUNG -- the live key
#       measured after the rung.
#   tests.phase33_state.GOLD_DERIVED_READINGS -- every reading and the
#       generation it was measured against, including this module's.
#
# The document-level assertions in this file read no gold and are not gated by
# anything. They run unconditionally, and they are what actually guards the
# document.
# ---------------------------------------------------------------------------
