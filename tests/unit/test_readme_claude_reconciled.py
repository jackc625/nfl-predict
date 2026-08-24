"""Permanent content guard for the repo-root README.md + CLAUDE.md reconciliation.

Phase 23 (Trust & Reproducibility) reconciled the two top-level narrative docs
to current v2.1 reality (D-06): README.md's stale Getting Started / Status /
Deployment / Project Structure / Current Limitations sections, and CLAUDE.md's
Project Overview / Current Status. A trust milestone whose front-door docs
reference deleted files or claim the wrong phase status undercuts the whole
point, so this committed test locks the contract:

- README.md uses the corrected setup command (``Copy-Item .env.example .env``),
  NOT the file deleted in Phase 19 (``deployment/production.env``);
- README.md no longer references any deleted file / non-existent Makefile target
  / deleted script (``make snapshot``, ``operational_monitoring.py``,
  ``validate_models.py``, ``make test-quick``, ``make dev-test``, the deleted
  Docker-stack artifact filenames, the deleted CI workflow);
- README.md cross-links MODEL-DIAGNOSIS.md for the production-vs-backtest
  mismatch;
- CLAUDE.md no longer claims "v1.0 MVP shipped 2026-03-28. 10 phases complete"
  as the current status, and references both v2.0 and v2.1.

Phase 30 (30-14) extended this guard to the Phase-30 end state in
TestPhase30Reconciled: both docs cross-link GATED-REFIT-READOUT.md, name all
three serving artifacts (including the two the gate REFUSED and therefore left
in place), drop the stale v2.1-is-current / 7-stage-PIPELINE / dynamic-blend-
pending claims, and attach no deployment verb to a refused candidate.

NOTE: README.md and CLAUDE.md predate the Windows cp1252 / ASCII-only convention
and legitimately contain non-ASCII characters (em-dashes, Unicode arrows). This
guard therefore does NOT assert ``content.isascii()`` -- it is a no-stale-
reference + corrected-string-presence contract only. (The new repo-root forensic
docs RUNBOOK.md / METHODOLOGY.md / STATE-OF-SYSTEM.md ARE ASCII-guarded by their
own committed tests.)

This is a permanent committed test, NOT a throwaway script.
"""

from pathlib import Path

# Repo root resolved from this file: tests/unit/test_readme_claude_reconciled.py -> repo root.
REPO_ROOT = Path(__file__).resolve().parents[2]
README_MD = REPO_ROOT / "README.md"
CLAUDE_MD = REPO_ROOT / "CLAUDE.md"


def _read(path: Path) -> str:
    """Read a repo-root markdown file."""
    return path.read_text(encoding="utf-8")


class TestReadmeReconciled:
    """README.md must use corrected commands and reference no deleted files."""

    def test_file_exists_at_repo_root(self):
        """README.md is present at the repo root."""
        assert README_MD.is_file(), f"missing: {README_MD}"

    def test_corrected_setup_command_present(self):
        """The corrected PowerShell setup command is documented."""
        content = _read(README_MD)
        assert "Copy-Item .env.example .env" in content

    def test_no_deleted_env_template_reference(self):
        """The Phase-19-deleted deployment/production.env is no longer referenced."""
        content = _read(README_MD)
        assert "deployment/production.env" not in content

    def test_no_nonexistent_make_targets(self):
        """Removed/non-existent Makefile targets are no longer documented."""
        content = _read(README_MD)
        for target in ("make snapshot", "make test-quick", "make dev-test"):
            assert target not in content, f"stale Makefile target present: {target}"

    def test_no_deleted_scripts_referenced(self):
        """Scripts deleted in Phase 19 are not referenced in the front door."""
        content = _read(README_MD)
        for script in ("operational_monitoring.py", "validate_models.py"):
            assert script not in content, f"deleted script referenced: {script}"

    def test_no_deleted_docker_stack_artifacts(self):
        """The deleted Docker/Nginx/CI artifact filenames are not referenced.

        The reconciled Deployment section may NAME the deleted stack while
        explaining it was removed, but it must not reference the concrete
        deleted artifact filenames as if they still exist.
        """
        content = _read(README_MD)
        for artifact in (
            "docker-compose.yml",
            "Dockerfile",
            "gunicorn.conf.py",
            "nginx.conf",
            "crontab.txt",
            "friday-production.yml",
        ):
            assert artifact not in content, f"deleted artifact referenced: {artifact}"

    def test_no_stale_phase_status(self):
        """The stale 'Phases 17-18 are not started' claim is gone (they shipped)."""
        content = _read(README_MD)
        assert "Phases 17-18 are not started" not in content

    def test_cross_links_model_diagnosis(self):
        """README cross-links MODEL-DIAGNOSIS.md for the prod-vs-backtest mismatch."""
        content = _read(README_MD)
        assert "MODEL-DIAGNOSIS.md" in content

    def test_no_stale_production_runs_v1_claim(self):
        """Phase 25 (D25-10): the present-tense 'production currently runs the v1.0
        pre-Elo models' / 'production continues to run the v1.0 WP/ATS models' claims
        are gone -- WP + ATS were activated through the gate, O/U retained.
        """
        content = _read(README_MD)
        stale_claims = (
            "Production currently\nruns the v1.0 pre-Elo models",
            "production\n   continues to run the v1.0 WP/ATS models",
            "production continues to run the v1.0 WP/ATS models",
        )
        present = [c for c in stale_claims if c in content]
        assert not present, (
            f"README.md still carries stale production-serves-v1 claims: {present}"
        )

    def test_cross_links_activation_readout(self):
        """README cross-links ACTIVATION-READOUT.md for the Phase-25 activation record."""
        content = _read(README_MD)
        assert "ACTIVATION-READOUT.md" in content, (
            "README.md should cross-link ACTIVATION-READOUT.md (the Phase-25 activation record)"
        )


class TestClaudeReconciled:
    """CLAUDE.md's Project Overview / Current Status must reflect v2.1 reality."""

    def test_file_exists_at_repo_root(self):
        """CLAUDE.md is present at the repo root."""
        assert CLAUDE_MD.is_file(), f"missing: {CLAUDE_MD}"

    def test_no_stale_current_status_claim(self):
        """CLAUDE.md no longer claims v1.0-MVP-only as the current status."""
        content = _read(CLAUDE_MD)
        assert "v1.0 MVP shipped 2026-03-28. 10 phases complete" not in content
        assert "**Current Status**: v1.0 MVP shipped. End-to-end" not in content

    def test_references_v2_0_and_v2_1(self):
        """CLAUDE.md references the shipped v2.0 and the in-progress v2.1 milestone."""
        content = _read(CLAUDE_MD)
        assert "v2.0" in content
        assert "v2.1" in content

    def test_silver_table_list_current(self):
        """The data-layers silver list names current tables, not stale team_stats."""
        content = _read(CLAUDE_MD)
        assert "weather, team_stats" not in content
        assert "team_form_features" in content

    def test_no_stale_production_serves_v1_claim(self):
        """Phase 25 (D25-10): the present-tense 'production currently serves the v1.0
        pre-Elo WP/ATS models' claim is gone -- WP + ATS were activated through the gate.

        The post-activation reality (WP/ATS serve re-fits; O/U retained v1.0) replaced it.
        A surviving 'production currently serves the v1.0 pre-Elo WP/ATS models' claim would
        be a false present-tense statement.
        """
        content = _read(CLAUDE_MD)
        assert (
            "production currently serves the v1.0 pre-Elo WP/ATS models" not in content
        ), (
            "CLAUDE.md still claims production serves v1.0 pre-Elo WP/ATS (false post Phase 25)"
        )

    def test_post_activation_reality_present(self):
        """CLAUDE.md states the post-activation reality + cross-links ACTIVATION-READOUT.md."""
        content = _read(CLAUDE_MD)
        assert "ACTIVATION-READOUT.md" in content, (
            "CLAUDE.md should cross-link ACTIVATION-READOUT.md for the activation record"
        )
        assert "v3.0" in content, (
            "CLAUDE.md should reference the in-progress v3.0 milestone"
        )
        assert "RETAINED" in content, (
            "CLAUDE.md should record that O/U retained v1.0 (the honest-refusal outcome)"
        )


class TestPhase30Reconciled:
    """Phase 30 (30-14): the two front-door docs describe the Phase-30 end state.

    Phase 30 rebuilt gold, ruled on three feature groups, and re-ran the
    per-target deploy gate: WP PASSED and was promoted; ATS and O/U FAILED and
    their incumbents were RETAINED. The Phase-25 record is kept BESIDE the new
    one, never overwritten (the supersede-in-place rule), so the assertions
    below are additive -- the Phase-25 markers guarded above still apply.

    What is guarded here:

    1. Both docs cross-link ``GATED-REFIT-READOUT.md``.
    2. Both docs name the Phase-30 production pointers, including the two that
       did NOT move. A reader must be able to see what a refused candidate left
       serving, not merely that a phase ran.
    3. The stale milestone-status claims are gone -- README no longer says v2.1
       is the current milestone or that Phase 23 is in progress, and no longer
       points at a 7-stage PIPELINE.md (it has 8 stages since Phase 25).
    4. The stale "dynamic blend is implemented but not yet activated" claim is
       gone; the dynamic blend has been live for all three targets since Phase 25.
    5. No over-claim word is attached to a REFUSED target. A refusal leaves an
       incumbent serving; calling that "deployed", "promoted", "activated" or
       "shipped" would misreport the phase's single production change as three.
    """

    def test_readme_cross_links_gated_refit_readout(self):
        """README cross-links GATED-REFIT-READOUT.md (the Phase-30 record)."""
        content = _read(README_MD)
        assert "GATED-REFIT-READOUT.md" in content, (
            "README.md should cross-link GATED-REFIT-READOUT.md (the Phase-30 record)"
        )

    def test_claude_cross_links_gated_refit_readout(self):
        """CLAUDE.md cross-links GATED-REFIT-READOUT.md (the Phase-30 record)."""
        content = _read(CLAUDE_MD)
        assert "GATED-REFIT-READOUT.md" in content, (
            "CLAUDE.md should cross-link GATED-REFIT-READOUT.md (the Phase-30 record)"
        )

    def test_both_docs_name_the_phase30_production_pointers(self):
        """Both docs name all three serving artifacts, including the two retained.

        ``wp_20260824_113325`` is the one pointer Phase 30 moved.
        ``ats_20260605_220128`` and ``ou_20260326_163930`` are what the two
        REFUSED targets left serving -- naming them is what makes the refusals
        legible rather than merely mentioned.
        """
        pointers = (
            "wp_20260824_113325",
            "ats_20260605_220128",
            "ou_20260326_163930",
        )
        for path, label in ((README_MD, "README.md"), (CLAUDE_MD, "CLAUDE.md")):
            content = _read(path)
            missing = [p for p in pointers if p not in content]
            assert not missing, (
                f"{label} missing Phase-30 end-state production pointers: {missing}"
            )

    def test_readme_stale_milestone_status_absent(self):
        """README no longer presents v2.1 / Phase 23 as the current work."""
        content = _read(README_MD)
        stale_claims = (
            'The current milestone, v2.1 "Trust & Reproducibility"',
            "Phase 23 (documentation, runbook, state-of-system) is in progress",
            "**In progress (v2.1 Trust & Reproducibility).**",
        )
        present = [c for c in stale_claims if c in content]
        assert not present, (
            f"README.md still presents v2.1 as the current milestone: {present} "
            "(v2.1 shipped 2026-06-01; v3.0 is in progress)"
        )

    def test_readme_no_stale_seven_stage_pipeline_reference(self):
        """README points at the 8-stage PIPELINE.md, not the pre-Phase-25 7-stage one."""
        content = _read(README_MD)
        assert "7-stage sequence" not in content, (
            "README.md still points at a 7-stage PIPELINE.md "
            "(Phase 25 inserted Promote; it has 8 stages)"
        )

    def test_readme_no_stale_dynamic_blend_pending_claim(self):
        """The 'dynamic blend is ready to activate' claim is gone (it is live)."""
        content = _read(README_MD)
        assert (
            "ready to\nactivate when the next retrain clears gating" not in content
        ), "README.md still claims the dynamic blend is pending activation"
        assert "ready to activate when the next retrain clears gating" not in content, (
            "README.md still claims the dynamic blend is pending activation "
            "(it has been live for all three targets since Phase 25)"
        )

    def test_no_over_claim_word_attached_to_a_refused_target(self):
        """Neither doc describes a REFUSED Phase-30 target as deployed.

        Scans each line that names a refused target's artifact version and fails
        if that same line carries a deployment verb. Line-scoped rather than
        document-scoped on purpose: both docs legitimately use "promoted" of WP,
        which actually was, and a document-wide substring search would false-
        positive on that. The point is the ASSOCIATION, not the vocabulary.
        """
        refused_versions = ("ats_20260824", "ou_20260824")
        over_claim_words = (
            "deployed",
            "promoted",
            "activated",
            "shipped",
            "swapped in",
        )
        for path, label in ((README_MD, "README.md"), (CLAUDE_MD, "CLAUDE.md")):
            for line in _read(path).splitlines():
                lowered = line.lower()
                if not any(v in lowered for v in refused_versions):
                    continue
                hits = [w for w in over_claim_words if w in lowered]
                assert not hits, (
                    f"{label} attaches over-claim word(s) {hits} to a REFUSED "
                    f"Phase-30 candidate on this line: {line.strip()!r}"
                )
