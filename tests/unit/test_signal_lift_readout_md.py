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


@pytest.mark.integration
class TestReadoutMatchesHarness:
    """The doc-to-harness validation: the doc's CURRENT ruling is the harness's ruling.

    SCOPE, and why it is not the point estimate (D29-06-02, owner decision). This guard used to
    assert that re-running the harness reproduced the committed +0.177334 situational-OU delta.
    That assumes gold is frozen. v3.0 rebuilds gold on purpose -- Phase 29 widened it, Phase 30
    re-fits on it, and upstream nflreadpy revised 2018-2024 play-by-play underneath both -- so the
    assumption is permanently false and the assertion was guaranteed to keep going red for
    reasons that are not drift. Re-anchoring it to each new measurement was explicitly considered
    and rejected: it rewrites a published record to match a moving input, and drifts again on the
    next rebuild. That reasoning stands and is NOT relaxed here.

    What IS permanent, and is asserted here: the ruling the readout records as CURRENT must be
    the ruling the harness returns. A point estimate moving with gold is expected; the D-05
    ruling flipping is exactly the thing a tripwire should catch, and it stays caught.

    WHAT CHANGED ON 2026-09-05, and why this is not a weakened assertion. The guard previously
    pinned KEEP, because KEEP was what the readout recorded. Plan 31-11's full gold rebuild
    corrected the LAR -> LA odds-key orphan, so 68 games inside this screen's own 2021-2024
    holdout stopped being graded against a fabricated 0.0 market line and their ATS / O/U labels
    changed. On those corrected labels the screen returns a D-05 veto: situational-OU
    -0.3203552582994336, and injury and snap veto too. The guard's own failure message said the
    right thing -- a flipped ruling is a real finding that must be reconciled in the doc rather
    than re-anchored -- so the doc was reconciled (Section 0b, with the flip, its cause and the
    2026-06-29 anchor left standing) and the guard follows the doc. The assertion did not get
    looser: it pins a ruling exactly as before, and a return to KEEP now fails exactly as loudly,
    because that too would be the doc and the harness disagreeing.

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

    def test_situational_ou_current_ruling_reproduces_from_harness(self) -> None:
        """The ruling the doc records as CURRENT still reproduces from the committed harness."""
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

        # The permanent invariant, unchanged in kind: the ruling the doc records as CURRENT
        # must be the ruling the harness returns. A moved point estimate is expected; a flipped
        # ruling is not. Since the 2026-09-05 corrected-label rebuild the current ruling is DROP.
        assert decision["keep"] is False, (
            f"situational screens KEEP again: {decision['reason']} "
            f"(situational-OU delta {cell['delta_mean']}). Section 0b of the readout records the "
            "CURRENT ruling as DROP on a D-05 OU veto. A flipped ruling is a real finding, not "
            "point-estimate drift, and must be reconciled in the doc rather than re-anchored -- "
            "in EITHER direction. Reconcile Section 0b; do not relax this assertion."
        )
        assert cell["veto"] is True, (
            "situational-OU no longer carries the D-05 veto the readout records in Section 0b "
            f"(delta {cell['delta_mean']}); reconcile the doc, do not relax this guard"
        )

        content = _read_readout()
        # The published 2026-06-29 Phase-28 record stays recorded rather than being overwritten.
        assert _HISTORICAL_ANCHOR in content, (
            "the doc must keep recording the 2026-06-29 situational-OU +0.177334 anchor"
        )
        assert "DRIFT RECORD" in content, (
            "the doc must carry the dated drift record explaining the divergence (D29-06-02)"
        )
        # ... and the 2026-09-05 reconciliation stays recorded beside it: the measured delta,
        # the sentence naming which reading is current, and the cause named explicitly rather
        # than softened to "a correction was applied" (the DEF-31-09 disclosure standard).
        assert _CURRENT_MEASURED_DELTA in content, (
            "the doc must keep recording the 2026-09-05 measured situational-OU delta "
            f"{_CURRENT_MEASURED_DELTA}"
        )
        assert _CURRENT_RULING_MARKER in content, (
            "the doc must state which ruling is CURRENT; the harness returns "
            f"{decision['reason']!r} and Section 0b must say so"
        )
        assert _CURRENT_CAUSE_MARKER in content, (
            "the doc must name the fabricated 0.0 market line as the cause, not merely report "
            "that a correction was applied (DEF-31-09)"
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
