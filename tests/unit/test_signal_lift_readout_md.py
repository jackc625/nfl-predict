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
asserts a load-bearing per-cell delta reproduces from the committed harness and appears in the
doc, so the doc cannot silently drift from the numbers (the Phase-26 doc-drift convention).

ASCII only, no emoji (CLAUDE.md).
"""

from __future__ import annotations

from pathlib import Path

import pytest

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


@pytest.mark.integration
class TestReadoutMatchesHarness:
    """The deeper doc-to-harness validation: the doc's numbers ARE the harness numbers.

    Runs ``run_signal_lift_screen`` for the (situational, OU) cell -- whose paired delta is
    independent of which other groups are screened (the baseline leg excludes ALL Phase-28
    columns) -- and asserts the reproduced delta agrees with and appears in the doc. The
    determinism guard in tests/integration guarantees stability; this ties the number to the
    committed prose so the doc cannot silently drift.
    """

    def test_situational_ou_delta_reproduces_from_harness(self) -> None:
        """The situational-OU +0.177334 delta reproduces and is recorded in the doc."""
        if not (_GOLD_OU_PATH.exists() and _ODDS_PATH.exists()):
            pytest.skip(
                f"Canonical gold/odds not present at {_GOLD_OU_PATH} / {_ODDS_PATH}"
            )

        import warnings

        import pandas as pd

        from backtest.signal_lift import run_signal_lift_screen

        gold = pd.read_parquet(_GOLD_OU_PATH)
        odds = pd.read_parquet(_ODDS_PATH)

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            result = run_signal_lift_screen(
                gold_by_target={"ou": gold},
                closing_odds_df=odds,
                targets=["ou"],
                groups=["situational"],
            )

        cell = result["groups"]["situational"]["per_target"]["ou"]
        delta = cell["delta_mean"]
        assert delta is not None
        assert abs(delta - 0.177334) < 5e-3, (
            f"harness situational-OU delta {delta} drifted from the doc anchor +0.177334"
        )

        content = _read_readout()
        assert "+0.177334" in content, (
            "the doc must record the situational-OU +0.177334 lift anchor"
        )
        # The harness keep/drop decision agrees with the doc's KEEP ruling.
        assert result["groups"]["situational"]["decision"]["keep"] is True
