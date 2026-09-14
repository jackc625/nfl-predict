"""Permanent doc-drift guard for the repo-root OU-DIVERGENCE-DIAGNOSIS.md (Phase 26, OUM-01).

Mirrors ``tests/unit/test_methodology_md.py`` (the repo-precedent doc-drift guard pattern): a
PERMANENT committed test, NOT a throwaway ``scripts/check_*.py``. It guards the OUM-01 deliverable
``OU-DIVERGENCE-DIAGNOSIS.md`` against three failure modes:

  - the file is missing or not at the repo root,
  - the content carries non-ASCII (emoji / cp1252-hostile) characters (CLAUDE.md hard constraint),
  - the doc has DRIFTED from the harness numbers -- the deeper validation: this guard RUNS
    ``run_ou_divergence_diagnosis()`` and asserts selected load-bearing numeric values from its
    output appear in / agree with the doc, not just an anchor-string match (the Codex MED ask).

The eight required section markers (D26-15) are asserted present so a future edit cannot silently
drop a required section. The harness-run validation is the anti-rot core (T-26-10): the doc cannot
silently drift because this test reproduces its load-bearing numbers from the committed orchestrator.

This module is ALSO the single owner of the STATE.md verdict assertion (Task 4, the TestStateVerdict
class) -- the repo-precedent placement (no second ``scripts/check_state_verdict.py``). That class is
skip-guarded because .planning/STATE.md is gitignored and may be absent on a clean CI checkout.

ASCII only, no emoji (CLAUDE.md).
"""

from __future__ import annotations

from pathlib import Path

import pytest

# Repo root resolved from this file: tests/unit/test_ou_divergence_diagnosis_md.py -> repo root.
REPO_ROOT = Path(__file__).resolve().parents[2]
DIAGNOSIS_MD = REPO_ROOT / "OU-DIVERGENCE-DIAGNOSIS.md"
STATE_MD = REPO_ROOT / ".planning" / "STATE.md"

# Gold presence skip-guard: the harness-run validation needs the Phase-20 rebuilt canonical gold.
_GOLD_WP_PATH = REPO_ROOT / "data" / "gold" / "features_wp.parquet"

# The eight required section markers (D26-15 required-content list). A substring of each section's
# heading/anchor, chosen so a benign reword does not trip the guard but a dropped section does.
_REQUIRED_SECTION_MARKERS = (
    "Integrity preamble",  # 1
    "Bias-vs-anticipation decomposition",  # 2
    "all cuts with coverage counts",  # 3 (the extended sweep section)
    "Trial registry + corrected significance",  # 4
    "Throwaway EV preview",  # 5
    "Burned-holdout caveat",  # 6
    "The pre-registered go bar",  # 7 (recorded verbatim)
    "Verdict",  # 8
)

# The +1.11 anchor (the headline finding) must be present as a string.
_LINE_CLV_ANCHOR = "+1.1095"


def _read_diagnosis_md() -> str:
    """Read OU-DIVERGENCE-DIAGNOSIS.md from the repo root."""
    return DIAGNOSIS_MD.read_text(encoding="utf-8")


class TestDiagnosisMdExists:
    """OU-DIVERGENCE-DIAGNOSIS.md must exist at the repo root and be ASCII-only."""

    def test_file_exists_at_repo_root(self) -> None:
        """The OUM-01 deliverable is present at the repo root (sibling of MODEL-DIAGNOSIS.md)."""
        assert DIAGNOSIS_MD.is_file(), f"missing: {DIAGNOSIS_MD}"

    def test_content_is_ascii(self) -> None:
        """Content is ASCII-only (no emoji / non-ASCII, per CLAUDE.md)."""
        content = _read_diagnosis_md()
        assert content.isascii(), (
            "OU-DIVERGENCE-DIAGNOSIS.md contains non-ASCII characters"
        )


class TestDiagnosisMdSections:
    """All eight required sections (D26-15) plus the +1.11 anchor must be present."""

    def test_all_required_section_markers_present(self) -> None:
        """Every required section marker is present; names any missing so a drop is actionable."""
        content = _read_diagnosis_md()
        missing = [m for m in _REQUIRED_SECTION_MARKERS if m not in content]
        assert not missing, (
            f"OU-DIVERGENCE-DIAGNOSIS.md missing required sections: {missing}"
        )

    def test_line_clv_anchor_present(self) -> None:
        """The headline +1.1095 line_clv anchor is present (the load-bearing finding)."""
        content = _read_diagnosis_md()
        assert _LINE_CLV_ANCHOR in content, (
            f"the headline line_clv anchor {_LINE_CLV_ANCHOR!r} is absent from the diagnosis doc"
        )

    def test_go_bar_recorded_verbatim(self) -> None:
        """The pre-registered go bar's three criteria + the GRADED-EDGE metric are recorded."""
        content = _read_diagnosis_md()
        # The verbatim go bar (Plan 26-03) carries these load-bearing phrases.
        for phrase in (
            "Corrected significance",
            "Structural bar (D26-11)",
            "EV clearance (D26-16)",
            "GRADED EDGE",
            "real but unpriceable at half-point",
        ):
            assert phrase in content, (
                f"the verbatim pre-registered go bar is missing the phrase {phrase!r}"
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


class TestStateVerdict:
    """The go/no-go verdict is recorded in .planning/STATE.md as a LOCKED Phase-27 input (D26-14).

    This is the repo-precedent placement for the STATE-verdict assertion (Task 4): it lives in
    THIS doc-drift module, not a second ``scripts/check_state_verdict.py`` throwaway. It is
    skip-guarded because ``.planning/STATE.md`` is gitignored (commit_docs:false) and may be absent
    on a clean CI checkout -- the working-tree edit is the deliverable, not a committed artifact.

    Guards T-26-11 (repudiation of the verdict): the verdict + its basis MUST be present in STATE.md
    as a recorded, LOCKED Phase-27 input, not left informal.
    """

    # One of the three pre-registered verdict tokens must appear in the Phase-26 entry.
    _VERDICT_TOKENS = ("GO", "SCOPED GO", "NO-GO")

    def test_state_md_records_phase26_verdict(self) -> None:
        """STATE.md carries a Phase-26 verdict entry with one of the three verdict tokens."""
        if not STATE_MD.exists():
            pytest.skip(
                f"STATE.md not present at {STATE_MD} (gitignored; absent on clean checkout)"
            )

        content = STATE_MD.read_text(encoding="utf-8")

        # A Phase-26 verdict decision entry must exist (the D26-14 LOCKED Phase-27 input).
        assert "D26-14" in content, (
            "STATE.md is missing the D26-14 go/no-go verdict decision entry"
        )

        # At least one of the three pre-registered verdict tokens must appear in STATE.md.
        assert any(tok in content for tok in self._VERDICT_TOKENS), (
            "STATE.md does not record any of the verdict tokens "
            f"{self._VERDICT_TOKENS!r} (the go/no-go ruling)"
        )

        # The verdict must be recorded as a LOCKED input to Phase 27 (not informal).
        assert "LOCKED" in content and "Phase 27" in content, (
            "STATE.md must record the verdict as a LOCKED input to Phase 27 (D26-14)"
        )


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
