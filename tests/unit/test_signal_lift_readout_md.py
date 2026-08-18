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
    """The doc-to-harness validation: the doc's RULING is the harness's ruling.

    SCOPE, and why it is not the point estimate (D29-06-02, owner decision). This guard used to
    assert that re-running the harness reproduced the committed +0.177334 situational-OU delta.
    That assumes gold is frozen. v3.0 rebuilds gold on purpose -- Phase 29 widened it, Phase 30
    re-fits on it, and upstream nflreadpy revised 2018-2024 play-by-play underneath both -- so the
    assumption is permanently false and the assertion was guaranteed to keep going red for
    reasons that are not drift. Re-anchoring it to each new measurement was explicitly considered
    and rejected: it rewrites a published record to match a moving input, and drifts again on the
    next rebuild.

    What IS permanent, and is asserted here: the recorded KEEP ruling must still reproduce. A
    point estimate moving with gold is expected; the D-05 ruling flipping is exactly the thing a
    tripwire should catch, and it stays caught. The measured divergence and its three-cause
    decomposition live in Section 0a of the readout, dated, beside the original numbers rather
    than replacing them.

    THE BASELINE MUST BE PINNED, and this test is why the pin exists. It used to call
    ``run_signal_lift_screen`` with the module-default ``baseline_exclude_groups`` and justify
    itself with "the baseline leg excludes ALL Phase-28 columns". That default is ``GROUPS``, a
    deny-list of three names, and it could not name Phase 29's fifteen ``line_movement`` columns --
    which therefore landed in the BASELINE leg. With them there the cell reads -0.195371, a D-05
    veto, against a recorded KEEP: a ruling flip produced purely by baseline composition. Going
    through ``screen_kwargs_for_phase(28)`` is the fix, and it is deliberately the same seam
    ``main()`` uses, so the guard runs exactly the invocation the CLI runs. A group registered by
    a later phase is pinned out of the Phase-28 baseline automatically.
    """

    def test_situational_ou_keep_ruling_reproduces_from_harness(self) -> None:
        """The recorded KEEP ruling still reproduces, and the doc still records its anchor."""
        if not (_GOLD_OU_PATH.exists() and _ODDS_PATH.exists()):
            pytest.skip(
                f"Canonical gold/odds not present at {_GOLD_OU_PATH} / {_ODDS_PATH}"
            )

        import warnings

        import pandas as pd

        from backtest.signal_lift import run_signal_lift_screen, screen_kwargs_for_phase

        gold = pd.read_parquet(_GOLD_OU_PATH)
        odds = pd.read_parquet(_ODDS_PATH)

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            result = run_signal_lift_screen(
                gold_by_target={"ou": gold},
                closing_odds_df=odds,
                targets=["ou"],
                groups=["situational"],
                **screen_kwargs_for_phase(28),
            )

        decision = result["groups"]["situational"]["decision"]
        cell = result["groups"]["situational"]["per_target"]["ou"]

        # The permanent invariant: the ruling the doc records must still be the ruling the
        # harness returns. A moved point estimate is expected; a flipped ruling is not.
        assert decision["keep"] is True, (
            f"situational no longer screens KEEP: {decision['reason']} "
            f"(situational-OU delta {cell['delta_mean']}). The readout records KEEP -- a flipped "
            "ruling is a real finding, not point-estimate drift, and must be reconciled in the "
            "doc rather than re-anchored."
        )
        assert cell["veto"] is False, "situational-OU must not carry a D-05 veto"

        content = _read_readout()
        # The published Phase-28 record stays recorded, with its measurement date and the
        # dated drift record that explains why re-running returns something else.
        assert "+0.177334" in content, (
            "the doc must keep recording the 2026-06-29 situational-OU +0.177334 anchor"
        )
        assert "DRIFT RECORD" in content, (
            "the doc must carry the dated drift record explaining the divergence (D29-06-02)"
        )

    def test_phase28_baseline_is_pinned_against_later_widening(self) -> None:
        """The Phase-28 baseline excludes EVERY registered group, not just the three names.

        The regression this pins: with ``GROUPS`` (three names) a group registered by a later
        phase lands in the Phase-28 baseline and silently re-defines what the recorded grid
        measured. Asserting membership rather than a literal tuple means registering a Phase-31
        group keeps this passing, while reverting the pin to ``GROUPS`` fails it.
        """
        from backtest import signal_lift

        pinned = signal_lift.screen_kwargs_for_phase(28)["baseline_exclude_groups"]

        assert set(pinned) == set(signal_lift._GROUP_PREDICATE), (
            "the Phase-28 baseline must exclude every registered signal group"
        )
        assert "line_movement" in pinned, (
            "Phase 29's line_movement columns must not sit in the Phase-28 baseline leg"
        )
        assert set(signal_lift.GROUPS) < set(pinned), (
            "the pin must be a strict superset of the three screened Phase-28 groups"
        )
