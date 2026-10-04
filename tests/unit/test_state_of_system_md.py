"""Permanent content check for the repo-root STATE-OF-SYSTEM.md.

STATE-OF-SYSTEM.md is the v2.1 "Trust & Reproducibility" capstone (DOC-03): the
single consolidated answer to "what can I trust, what did we fix, what is still
deferred?" It is structured Trustworthy now / Fixed this milestone / Deferred,
where the Deferred section is the SINGLE registry of items otherwise scattered
across five docs -- each a one-line pointer, NOT a re-assertion. This committed
test guards the required section anchors + the hard cross-links so the capstone
cannot silently drift:

- the file exists at the repo root,
- the content is ASCII-only (no emoji, per CLAUDE.md / the Windows cp1252
  constraint),
- the three section anchors are present (Trustworthy / Fixed / Deferred),
- the DOC-03 HARD link to MODEL-DIAGNOSIS.md is present (the DIAG accuracy
  report DOC-03 names explicitly), and
- the AUDIT-REPORT.md + AUTOMATION.md links (the Trustworthy/Fixed/Deferred
  source docs) are present.

The docs/ removal assertion and the methodology negative assertions live in
test_methodology_md.py, NOT here.

This is a permanent committed test, NOT a throwaway script. It intentionally
FAILS until STATE-OF-SYSTEM.md is written (the Wave-2 doc commit turns it GREEN).
"""

from pathlib import Path

# Repo root resolved from this file: tests/unit/test_state_of_system_md.py -> repo root.
REPO_ROOT = Path(__file__).resolve().parents[2]
STATE_OF_SYSTEM_MD = REPO_ROOT / "docs" / "guides" / "STATE-OF-SYSTEM.md"


def _read_state_of_system_md() -> str:
    """Read STATE-OF-SYSTEM.md from the repo root."""
    return STATE_OF_SYSTEM_MD.read_text(encoding="utf-8")


class TestStateOfSystemMdExists:
    """STATE-OF-SYSTEM.md must exist at the repo root and be ASCII-only."""

    def test_file_exists_at_repo_root(self):
        """STATE-OF-SYSTEM.md is present at the repo root."""
        assert STATE_OF_SYSTEM_MD.is_file(), f"missing: {STATE_OF_SYSTEM_MD}"

    def test_content_is_ascii(self):
        """Content is ASCII-only (no emoji / non-ASCII, per CLAUDE.md)."""
        content = _read_state_of_system_md()
        assert content.isascii(), "STATE-OF-SYSTEM.md contains non-ASCII characters"


class TestStateOfSystemMdSections:
    """The three capstone section anchors must be present (DOC-03)."""

    def test_documents_trustworthy_fixed_deferred(self):
        """The Trustworthy / Fixed / Deferred section anchors are present.

        Names the missing anchors so a drift is actionable.
        """
        content = _read_state_of_system_md()
        anchors = ("Trustworthy", "Fixed", "Deferred")
        missing = [anchor for anchor in anchors if anchor not in content]
        assert not missing, f"STATE-OF-SYSTEM.md missing section anchors: {missing}"


class TestStateOfSystemMdLinks:
    """The DOC-03 hard link + the Trustworthy/Fixed/Deferred source links."""

    def test_links_model_diagnosis(self):
        """DOC-03 HARD requirement: STATE-OF-SYSTEM.md MUST link MODEL-DIAGNOSIS.md.

        This is the DIAG accuracy report DOC-03 names explicitly; the link is the
        single non-negotiable cross-reference for this doc.
        """
        content = _read_state_of_system_md()
        assert "MODEL-DIAGNOSIS.md" in content, (
            "DOC-03 requires a MODEL-DIAGNOSIS.md link"
        )

    def test_links_audit_report(self):
        """The Trustworthy/Fixed source doc (AUDIT-REPORT.md) is linked."""
        content = _read_state_of_system_md()
        assert "AUDIT-REPORT.md" in content

    def test_links_automation(self):
        """The deferred ALERT-WIRING source doc (AUTOMATION.md) is linked."""
        content = _read_state_of_system_md()
        assert "AUTOMATION.md" in content


class TestStateOfSystemMdActivationReconciled:
    """Phase 25 (D25-10): the DIAG-05 gated-re-fit RECOMMENDATION status is reconciled.

    The DIAG-05 gated re-fit was a Deferred future-milestone recommendation in v2.1; v3.0
    Phase 25 executed it (WP + ATS activated through the gate, O/U retained). This guards
    that the registry records the recommendation as EXECUTED and links the activation record
    (a recommendation-status update, not a literal 'pre-Elo' string swap -- no such literal
    is present in this doc).
    """

    def test_diag05_recommendation_marked_executed(self):
        """The DIAG-05 gated re-fit is recorded as EXECUTED, not still merely deferred."""
        content = _read_state_of_system_md()
        assert "EXECUTED in v3.0 Phase 25" in content, (
            "STATE-OF-SYSTEM.md should record the DIAG-05 gated re-fit as EXECUTED in Phase 25"
        )

    def test_links_activation_readout(self):
        """The Phase-25 activation record (ACTIVATION-READOUT.md) is linked."""
        content = _read_state_of_system_md()
        assert "ACTIVATION-READOUT.md" in content, (
            "STATE-OF-SYSTEM.md should link ACTIVATION-READOUT.md (the Phase-25 activation)"
        )


class TestStateOfSystemMdPhase30Reconciled:
    """Phase 30 (30-14): the registry records what Phase 30 closed and what stayed open.

    This registry is the SINGLE consolidated open list, so it is the one place where
    "is this still a problem?" gets answered. Two failure modes are guarded:

    1. An item Phase 30 CLOSED still being listed as deferred, which sends a future
       reader to fix something already fixed.
    2. An item Phase 30 LEFT OPEN quietly disappearing, which is the worse of the two --
       it makes the phase look cleaner than it was. The seven quarantines, the six
       deferred registers and the two deliberate residuals are all asserted present.
    """

    def test_links_gated_refit_readout(self):
        """The Phase-30 gated re-fit record is linked."""
        content = _read_state_of_system_md()
        assert "GATED-REFIT-READOUT.md" in content, (
            "STATE-OF-SYSTEM.md should link GATED-REFIT-READOUT.md (the Phase-30 record)"
        )

    def test_closed_items_no_longer_listed_as_deferred(self):
        """The two Deferred entries Phase 30 closed are gone from the registry.

        Both were listed under 'Deferred' and both are now fixed in code/data:
        the 2025 trailing-coverage backfill (gold's 2025 slice is now weeks 1-22),
        and the build_features --save no-op (a real --no-save exists).
        """
        content = _read_state_of_system_md()
        stale_deferred = (
            "2025 ingested through ~wk4 then frozen; backfill deferred to the",
            "so the flag is a no-op and the gold-write cannot be disabled via CLI",
        )
        present = [c for c in stale_deferred if c in content]
        assert not present, (
            f"STATE-OF-SYSTEM.md still defers items Phase 30 closed: {present}"
        )

    def test_records_the_two_deliberate_residuals(self):
        """The two residuals Phase 30 deliberately left open are NAMED, not dropped.

        (a) the within-season lookahead the prior-seasons-only bounds fix does not
        close, and (b) the dropped group's columns remaining physically in gold with
        no marker recording the exclusion. A residual that is not written down is
        indistinguishable from one that was never noticed.
        """
        content = _read_state_of_system_md()
        assert "within-season" in content, (
            "STATE-OF-SYSTEM.md should name the accepted within-season lookahead residual "
            "(the prior-seasons-only bounds fix closes the cross-season case only)"
        )
        assert "remain physically in gold" in content, (
            "STATE-OF-SYSTEM.md should record that the DROPPED group's columns remain "
            "physically in gold and are excluded at train time, with no marker in gold"
        )

    def test_records_the_seven_open_quarantines(self):
        """The seven still-open quarantined reproductions are recorded with their proof."""
        content = _read_state_of_system_md()
        assert "7 xfailed" in content, (
            "STATE-OF-SYSTEM.md should cite the suite's standing '7 xfailed' as the "
            "mechanical proof that seven quarantined reproductions are still open"
        )

    def test_records_the_six_open_deferred_registers(self):
        """Each of the six still-open Phase-30 registers is named individually."""
        content = _read_state_of_system_md()
        registers = (
            "D30-DEFER-04",
            "D30-DEFER-17",
            "D30-DEFER-22",
            "D30-DEFER-23",
            "D30-DEFER-24",
            "D30-DEFER-25",
        )
        missing = [r for r in registers if r not in content]
        assert not missing, (
            f"STATE-OF-SYSTEM.md missing still-OPEN Phase-30 registers: {missing}"
        )

    def test_ou_headline_clv_hazard_is_not_published_as_a_clv(self):
        """The +45.81 export figure never appears without its warning.

        ``BacktestResults.headline_clv`` is the mean of ``probability_clv`` for every
        target, and for O/U that column is not a line-CLV. Quoting +45.81 as an O/U
        CLV overstates a quantity that does not exist by roughly 40x. The registry may
        MENTION the figure -- that is the point of a reading hazard -- but only beside
        the true line_clv mean.
        """
        content = _read_state_of_system_md()
        if "45.81" in content:
            assert "1.8954569" in content, (
                "STATE-OF-SYSTEM.md quotes the +45.81 export figure without the true "
                "O/U line_clv mean (+1.8954569) beside it"
            )
            assert "NOT a CLV" in content, (
                "STATE-OF-SYSTEM.md quotes +45.81 without stating it is NOT a CLV"
            )


class TestPhase31Reconciled:
    """Phase 31 (31-19): the consolidated open list describes the milestone-close end state.

    The lockstep rule this repository adopted after a late drift failure: a document and its
    content-drift guard move in the SAME commit. What is guarded here:

    1. The registry cross-links ``PROFITABILITY-READOUT.md``, the milestone close.
    2. It records that Phase 31 deployed NO model. A productization phase that spent the single
       clean split is easy to misremember as a phase that changed production; it did not, and the
       distinction is what makes the measurement a measurement of what is actually serving.
    3. It names the four still-open items Phase 31 ADDS, rather than reporting a clean close.
    4. The standing quarantine and deferred-register counts are CARRIED FORWARD UNCHANGED. This is
       the assertion that stops a later phase from quietly absorbing an open item: the literal
       ``7 xfailed`` proof is already pinned by ``test_records_the_seven_open_quarantines``, and
       this class asserts Phase 31 did not claim to have closed any of them.
    """

    def test_cross_links_the_profitability_readout(self):
        """The milestone close is reachable from the single consolidated open list."""
        content = _read_state_of_system_md()
        assert "PROFITABILITY-READOUT.md" in content, (
            "STATE-OF-SYSTEM.md should cross-link PROFITABILITY-READOUT.md (the Phase-31 "
            "milestone close). The registry is the entry point for 'what can I trust now'."
        )

    def test_records_that_phase_31_deployed_no_model(self):
        """A productization phase that spent the clean split changed nothing in production."""
        content = _read_state_of_system_md()
        assert "deployed no model" in content.lower(), (
            "STATE-OF-SYSTEM.md does not record that Phase 31 deployed no model. Without it a "
            "reader cannot tell whether the 2025 verdict measured what is serving or something "
            "the phase had just changed underneath it."
        )
        assert "artifacts/latest.json` is byte-unchanged" in content, (
            "STATE-OF-SYSTEM.md does not record that the production swap surface is "
            "byte-unchanged by Phase 31"
        )

    def test_records_the_four_items_phase_31_adds(self):
        """Each new open item is named, not summarized away."""
        content = _read_state_of_system_md()
        markers = {
            "the renamed edge band": "three incompatible units",
            "the api/cache sign defect": "sign defect",
            "the non-computable forward CLV": "NOT COMPUTABLE",
            "the stale betting ledger": "has not been regenerated since",
        }
        missing = [name for name, marker in markers.items() if marker not in content]
        assert not missing, (
            f"STATE-OF-SYSTEM.md does not record these Phase-31 open items: {missing}. The "
            "registry is the SINGLE consolidated open list; an item that is not here is an item "
            "nobody will find."
        )

    def test_phase_31_does_not_claim_to_close_any_standing_quarantine(self):
        """The counts are carried forward unchanged, which is the point of a standing count."""
        content = _read_state_of_system_md()
        assert "Nothing above is closed by Phase 31" in content, (
            "STATE-OF-SYSTEM.md does not state that Phase 31 closes none of the standing open "
            "items. A milestone-close section that is silent on this reads as a clean close."
        )
        assert "remain OPEN and their counts are carried forward" in content, (
            "STATE-OF-SYSTEM.md does not state that the standing quarantine and "
            "deferred-register counts are carried forward unchanged"
        )

    def test_the_higher_whole_suite_xfail_count_is_explained_not_hidden(self):
        """The standing 7 is a Phase-30 count; the whole suite is higher, and that is stated.

        Without this sentence a reader who runs the suite, counts the xfails and finds more than
        seven has to choose between two bad conclusions: that the registry is wrong, or that
        something was unquarantined by accident. It is neither.
        """
        content = _read_state_of_system_md()
        assert "whole-suite xfailed count is HIGHER" in content, (
            "STATE-OF-SYSTEM.md does not explain why the whole-suite xfailed count exceeds the "
            "standing Phase-30 quarantine count of seven"
        )

    def test_the_five_deliberately_red_tests_are_pointed_at(self):
        """An operator seeing five reds must be able to find out why before concluding anything."""
        content = _read_state_of_system_md()
        assert "DELIBERATELY RED" in content, (
            "STATE-OF-SYSTEM.md does not record that five tests are deliberately red"
        )
        assert "PROFITABILITY-READOUT.md` section 8" in content, (
            "STATE-OF-SYSTEM.md does not point at the section explaining the five red tests"
        )
