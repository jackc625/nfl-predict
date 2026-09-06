"""Permanent content check for the repo-root METHODOLOGY.md + the docs/ removal.

METHODOLOGY.md (D-04) is the consolidated, de-staled portfolio deep-dive that
merges the still-true core of the two v1.0-era docs (docs/model_methodology.md +
docs/feature_engineering.md) and CROSS-LINKS current code rather than re-asserting
~1,274 stale lines. This committed test guards:

- the file exists at the repo root,
- the content is ASCII-only (no emoji, per CLAUDE.md / the Windows cp1252
  constraint),
- the current-code cross-links are present (models/, features/, ratings/elo.py),
- the drift landmines are ABSENT: no "placeholder 1500" Elo claim (Elo was
  activated in Phase 11), and no "RandomizedSearchCV" claim (replaced by Optuna
  in Phase 12).

This test file is ALSO the single owner of the docs/ removal assertion (D-01/D-04):
the six stale docs/*.md are deleted and the now-empty docs/ directory is removed.
That check lives in a separate TestStaleDocsRemoved class so it reads independently
of the METHODOLOGY content checks.

This is a permanent committed test, NOT a throwaway script. It intentionally FAILS
until METHODOLOGY.md is written AND the six docs/*.md are deleted (the Wave-2 doc
commit + the deletion turn it GREEN).
"""

from pathlib import Path

# Repo root resolved from this file: tests/unit/test_methodology_md.py -> repo root.
REPO_ROOT = Path(__file__).resolve().parents[2]
METHODOLOGY_MD = REPO_ROOT / "METHODOLOGY.md"


def _read_methodology_md() -> str:
    """Read METHODOLOGY.md from the repo root."""
    return METHODOLOGY_MD.read_text(encoding="utf-8")


class TestMethodologyMdExists:
    """METHODOLOGY.md must exist at the repo root and be ASCII-only."""

    def test_file_exists_at_repo_root(self):
        """METHODOLOGY.md is present at the repo root."""
        assert METHODOLOGY_MD.is_file(), f"missing: {METHODOLOGY_MD}"

    def test_content_is_ascii(self):
        """Content is ASCII-only (no emoji / non-ASCII, per CLAUDE.md)."""
        content = _read_methodology_md()
        assert content.isascii(), "METHODOLOGY.md contains non-ASCII characters"


class TestMethodologyMdCrossLinks:
    """The current-code cross-links (D-04) must be present."""

    def test_cross_links_current_code(self):
        """The methodology cross-links current code rather than re-asserting prose.

        Names the missing cross-links so a drift is actionable.
        """
        content = _read_methodology_md()
        anchors = ("models/", "features/", "ratings/elo.py")
        missing = [anchor for anchor in anchors if anchor not in content]
        assert not missing, f"METHODOLOGY.md missing code cross-links: {missing}"


class TestMethodologyMdNoDriftLandmines:
    """The v1.0-era drift landmines must NOT appear (D-04 still-true filter)."""

    def test_no_placeholder_elo_claim(self):
        """No 'placeholder 1500' Elo claim (Elo activated in Phase 11).

        The v1.0 docs described placeholder 1500.0 Elo; production now runs real
        Elo (6263 per-game snapshots). Any surviving placeholder-Elo prose is a
        false claim. Guard the conjunction so a benign mention of either word
        alone does not trip the test.
        """
        lowered = _read_methodology_md().lower()
        placeholder_elo_claim = "placeholder" in lowered and "1500" in lowered
        assert not placeholder_elo_claim, (
            "METHODOLOGY.md still carries the stale 'placeholder 1500' Elo claim"
        )

    def test_no_randomized_search_cv_claim(self):
        """No 'RandomizedSearchCV' claim (replaced by Optuna in Phase 12)."""
        lowered = _read_methodology_md().lower()
        assert "randomizedsearchcv" not in lowered, (
            "METHODOLOGY.md still references RandomizedSearchCV (replaced by Optuna)"
        )

    def test_no_stale_production_serves_v1_claim(self):
        """Phase 25 (D25-10): the present-tense 'production still serves the v1.0 pre-Elo
        artifacts' claim is gone -- WP + ATS were activated through the gate, O/U retained.
        """
        content = _read_methodology_md()
        assert "production still serves the v1.0 pre-Elo" not in content, (
            "METHODOLOGY.md still claims production serves v1.0 pre-Elo (false post Phase 25)"
        )
        assert "the deployed artifacts are still the v1.0 ones" not in content, (
            "METHODOLOGY.md landmine 3 still claims the deployed artifacts are all v1.0"
        )


class TestMethodologyMdActivationReconciled:
    """Phase 25 (D25-10): METHODOLOGY reflects the post-activation mixed deployed set."""

    def test_cross_links_activation_readout(self):
        """METHODOLOGY cross-links ACTIVATION-READOUT.md for the Phase-25 activation."""
        content = _read_methodology_md()
        assert "ACTIVATION-READOUT.md" in content, (
            "METHODOLOGY.md should cross-link ACTIVATION-READOUT.md (the activation record)"
        )

    def test_records_retained_ou(self):
        """METHODOLOGY records the honest-refusal outcome (O/U retained on v1.0)."""
        content = _read_methodology_md()
        assert "RETAINED" in content, (
            "METHODOLOGY.md should record that O/U retained v1.0 (the honest-refusal outcome)"
        )


class TestMethodologyMdPhase30Method:
    """Phase 30 (30-14): the methodology record carries the two-stage decision unit.

    Phase 30 added the project's most methodologically substantial machinery, and a
    deep-dive that omits it is stale in the way that matters most for this document --
    it would describe a per-target gate as the whole decision when the decision is now
    two stages answering two different questions. Guarded here:

    1. The two-stage decision unit is described (a per-GROUP selection rule feeding a
       per-TARGET deploy gate), because the split is what stops a group's cost on one
       target riding into production on another target's benefit.
    2. The pre-registration discipline is described, INCLUDING that its ordering is
       checked from git rather than asserted in prose. "The rule predates the result"
       is only worth writing down if it is checkable.
    3. The full-grid correction and its denominator are described.
    4. The three-valued verdict vocabulary is present and UNDETERMINED is stated as
       reported-not-collapsed. Merging "it hurt" with "we could not tell" destroys the
       finding a later phase needs.
    5. The prior-seasons-only bounds are described WITH their accepted within-season
       residual. Describing the fix without the residual would overstate it.
    """

    def test_cross_links_gated_refit_readout(self):
        """METHODOLOGY cross-links GATED-REFIT-READOUT.md (the Phase-30 record)."""
        content = _read_methodology_md()
        assert "GATED-REFIT-READOUT.md" in content, (
            "METHODOLOGY.md should cross-link GATED-REFIT-READOUT.md (the Phase-30 record)"
        )

    def test_describes_the_two_stage_decision_unit(self):
        """The per-GROUP selection rule and the per-TARGET deploy gate are both named."""
        content = _read_methodology_md()
        assert "two-stage decision unit" in content, (
            "METHODOLOGY.md should describe the Phase-30 two-stage decision unit"
        )
        assert "per-GROUP selection rule" in content, (
            "METHODOLOGY.md should name Stage 1 as a per-GROUP selection rule"
        )
        assert "per-TARGET deploy gate" in content, (
            "METHODOLOGY.md should name Stage 2 as the per-TARGET deploy gate"
        )

    def test_describes_pre_registration_checked_from_git(self):
        """The pre-registration is described as a git-checkable ancestry relation."""
        content = _read_methodology_md()
        lowered = content.lower()
        assert "pre-registration" in lowered or "pre-registered" in lowered, (
            "METHODOLOGY.md should describe the Phase-30 pre-registration discipline"
        )
        assert "ancestry" in lowered, (
            "METHODOLOGY.md should state that the pre-registration's ordering is an "
            "ANCESTRY relation checked from git, not a claim asserted in prose"
        )

    def test_describes_the_full_grid_correction_and_its_denominator(self):
        """The correction method, its family and the realized denominator are described."""
        content = _read_methodology_md()
        assert "Benjamini-Hochberg" in content, (
            "METHODOLOGY.md should name the Benjamini-Hochberg correction"
        )
        assert "m = 6" in content, (
            "METHODOLOGY.md should state the REALIZED denominator (m = 6, not 9) and that "
            "the exclusions made the surviving family easier to reject in, not harder"
        )

    def test_describes_the_three_valued_verdict_vocabulary(self):
        """All four verdict words appear and UNDETERMINED is reported, not collapsed."""
        content = _read_methodology_md()
        for verdict in ("KEEP", "DROP", "UNDETERMINED", "NOT MEASURED"):
            assert verdict in content, (
                f"METHODOLOGY.md missing verdict-vocabulary term: {verdict}"
            )
        assert "never collapsed into DROP" in content, (
            "METHODOLOGY.md should state that UNDETERMINED is reported as UNDETERMINED "
            "and never collapsed into DROP"
        )

    def test_describes_prior_seasons_bounds_with_their_residual(self):
        """The bounds fix is described together with the residual it does NOT close."""
        content = _read_methodology_md()
        assert (
            "strictly prior seasons" in content or "strictly-prior seasons" in content
        ), (
            "METHODOLOGY.md should describe the prior-seasons-only imputation/winsorization "
            "bounds introduced in Phase 30"
        )
        assert "within-season" in content, (
            "METHODOLOGY.md should state the accepted within-season residual the "
            "prior-seasons-only bounds fix does NOT close"
        )

    def test_no_stale_ou_only_dynamic_blend_claim(self):
        """The 'only O/U dynamic blend is adopted' claim is gone.

        The deployed blend artifact runs dynamic for all three targets. The stale
        claim contradicted the production manifest, which is the kind of drift this
        guard exists to catch.
        """
        content = _read_methodology_md()
        stale_claims = (
            "Only the O/U\n  dynamic blend is ADOPTED",
            "Only the O/U dynamic blend is ADOPTED",
            "**Dynamic blend is gated per target (only O/U adopted).**",
        )
        present = [c for c in stale_claims if c in content]
        assert not present, (
            f"METHODOLOGY.md still claims only O/U runs the dynamic blend: {present} "
            "(the deployed artifact runs dynamic for all three targets)"
        )


class TestStaleDocsRemoved:
    """The six stale docs/*.md and the docs/ directory must be gone (D-01/D-04).

    This class is the SINGLE owner of the docs/ removal assertion across the
    Phase 23 doc guards.
    """

    def test_six_stale_docs_deleted(self):
        """Each of the six stale docs/*.md no longer exists.

        Names every surviving file so the deletion gap is actionable.
        """
        basenames = (
            "operational_runbooks",
            "operational_procedures",
            "GETTING_STARTED",
            "api_documentation",
            "feature_engineering",
            "model_methodology",
        )
        surviving = [
            name for name in basenames if (REPO_ROOT / "docs" / f"{name}.md").exists()
        ]
        assert not surviving, f"stale docs/*.md still present: {surviving}"

    def test_docs_directory_removed(self):
        """The now-empty docs/ directory is removed (D-04)."""
        assert not (REPO_ROOT / "docs").exists(), "docs/ directory still present"


class TestPhase31MethodReconciled:
    """Phase 31 (31-19): METHODOLOGY.md carries the one-shot profitability method.

    Paired with METHODOLOGY.md in the same commit. Section 10 documents the two-stage DEPLOY
    decision; it says nothing about whether anything makes money, and the instrument that answers
    that is different in kind. The three things a reader has to be able to find here are the
    three chains, why the design is one-shot, and the verdict vocabulary -- because a reader who
    does not know the split is single-use will assume the number can simply be re-run.
    """

    def test_describes_the_three_per_target_ev_chains(self):
        """One bet decision source, three chains, and the price each target is struck at."""
        content = _read_methodology_md()
        for marker in ("BetSelector.select", "quarter Kelly", "devigging"):
            assert marker in content, (
                f"METHODOLOGY.md does not describe {marker!r} as part of the per-target EV "
                "chains. A method document that omits how a bet is priced and sized cannot be "
                "used to check the verdict that rests on it."
            )

    def test_describes_the_one_shot_design_and_its_refusals(self):
        """Irreversible and unrepeatable by construction, or it becomes another burned split."""
        content = _read_methodology_md()
        for marker in ("ONE-SHOT", "EXCLUSIVE file creation", "no force flag"):
            assert marker in content, (
                f"METHODOLOGY.md does not describe {marker!r}. The one-shot ledger is what makes "
                "the 2025 result a clean out-of-sample measurement rather than a fourth look at "
                "a holdout."
            )

    def test_states_the_cost_of_the_one_shot_design(self):
        """The trade is stated rather than discovered: the result cannot be re-run to check it."""
        content = _read_methodology_md()
        assert "cannot be checked by re-running it" in content, (
            "METHODOLOGY.md does not state the cost of the one-shot design. A method that cannot "
            "be re-run has a different verification story, and hiding that is how a consistency "
            "guard gets mistaken for a reproduction guard."
        )

    def test_describes_the_five_valued_verdict_vocabulary(self):
        """A zero-bet chain reports a RESULT, and a positive-but-insignificant one is not profit."""
        content = _read_methodology_md()
        for token in (
            "PROFITABLE_CLEAN",
            "UNPROFITABLE_CLEAN",
            "INCONCLUSIVE_CLEAN",
            "UNDISCHARGEABLE_NO_BETS",
            "UNDISCHARGEABLE_NO_CHAIN",
        ):
            assert token in content, (
                f"METHODOLOGY.md does not declare the verdict token {token!r}. The vocabulary is "
                "closed and was fixed before the numbers existed; a document that lists only the "
                "tokens that fired cannot show that."
            )
        assert "NEVER called profitable" in content, (
            "METHODOLOGY.md does not state that a positive return failing its p-value test is "
            "never called profitable"
        )

    def test_keeps_clv_and_roi_apart_as_different_tests(self):
        """The conflation this milestone exists to refuse, stated in the method document."""
        content = _read_methodology_md()
        assert "REPORT-ONLY" in content, (
            "METHODOLOGY.md does not state that CLV is report-only in the profitability chain"
        )
        assert "different tests of different quantities" in content, (
            "METHODOLOGY.md does not state that CLV and ROI are different tests of different "
            "quantities. A method document that leaves that implicit invites the exact misreading "
            "Phase 26 diagnosed."
        )

    def test_cross_links_the_profitability_readout(self):
        """The method points at the record it produced."""
        content = _read_methodology_md()
        assert "PROFITABILITY-READOUT.md" in content, (
            "METHODOLOGY.md should cross-link PROFITABILITY-READOUT.md"
        )
