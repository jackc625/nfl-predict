"""Permanent content check for the repo-root RUNBOOK.md.

RUNBOOK.md is the operator-task layer (DOC-01/DOC-02): how to perform every
common operation, how to tell whether a run succeeded, troubleshooting, recovery,
and an Architecture & Data-Flow section. It BUILDS ON the already-accurate
PIPELINE.md (commands) and AUTOMATION.md (Friday automation) rather than
duplicating them. This committed test guards the required section anchors so the
doc cannot silently drift out of sync with the code/docs it points at:

- the file exists at the repo root,
- the content is ASCII-only (no emoji, per CLAUDE.md / the Windows cp1252
  constraint),
- all 11 common operations are documented (ingest, build features, train,
  promote, backtest, predict, build-cache, serve, rollback, run automation,
  rebuild gold),
- the PIPELINE.md / AUTOMATION.md / README.md cross-references are present
  (link-don't-duplicate: PIPELINE = commands, AUTOMATION = automation,
  README = the architecture diagram for the DOC-02 section),
- a DOC-02 Architecture & Data-Flow section anchor is present,
- the D-02 per-command verification-basis labels are present (both the live
  label "verified live" AND the audit label "AUDIT-01"), and
- no stale reference to a Phase-19-deleted file/script survives
  (deployment/production.env, make snapshot, operational_monitoring.py,
  validate_models.py).

This is a permanent committed test, NOT a throwaway script. It intentionally
FAILS until RUNBOOK.md is written (the Wave-2 doc commit turns it GREEN).
"""

from pathlib import Path

# Repo root resolved from this file: tests/unit/test_runbook_md.py -> repo root.
REPO_ROOT = Path(__file__).resolve().parents[2]
RUNBOOK_MD = REPO_ROOT / "RUNBOOK.md"


def _read_runbook_md() -> str:
    """Read RUNBOOK.md from the repo root."""
    return RUNBOOK_MD.read_text(encoding="utf-8")


class TestRunbookMdExists:
    """RUNBOOK.md must exist at the repo root and be ASCII-only."""

    def test_file_exists_at_repo_root(self):
        """RUNBOOK.md is present at the repo root."""
        assert RUNBOOK_MD.is_file(), f"missing: {RUNBOOK_MD}"

    def test_content_is_ascii(self):
        """Content is ASCII-only (no emoji / non-ASCII, per CLAUDE.md)."""
        content = _read_runbook_md()
        assert content.isascii(), "RUNBOOK.md contains non-ASCII characters"


class TestRunbookMdOperations:
    """All 11 common operations (DOC-01) must be documented.

    Phase 25 (D25-16/17) inserted two operations to match the 8-stage PIPELINE
    order and the new rollback path: Promote (operation 4, between Train and
    Backtest) and Rollback (operation 9, reversing a Promote). Run automation
    renumbered from 8 to 10. Phase 30 (30-14) APPENDED an eleventh, Rebuild gold
    -- appended rather than inserted precisely so the ten existing heading
    anchors below do not renumber (Pitfall 5).
    """

    def test_documents_all_eleven_operations(self):
        """Each of the 11 operations is documented by its numbered section heading.

        Anchoring on the ``### N.`` operation headings (instead of a bare verb
        like ``backtest`` that also appears in cross-references, the architecture
        table, and troubleshooting) means deleting or gutting an operation
        SECTION is actually caught -- a generic substring could pass even with the
        section removed. Names the missing operations so a drift is actionable.
        """
        content = _read_runbook_md()
        operation_headings = (
            "### 1. Ingest",
            "### 2. Build features",
            "### 3. Train",
            "### 4. Promote models",
            "### 5. Backtest",
            "### 6. Predict",
            "### 7. Build cache",
            "### 8. Serve",
            "### 9. Rollback",
            "### 10. Run automation",
            "### 11. Rebuild gold",
        )
        missing = [heading for heading in operation_headings if heading not in content]
        assert not missing, f"RUNBOOK.md missing operation sections: {missing}"

    def test_documents_operation_canonical_commands(self):
        """Each operation section carries its canonical script/command token.

        Complements the heading anchor: the heading proves the section exists,
        the command token proves it still documents the real operation (not an
        empty stub). These are the unique script/command tokens, not bare verbs.
        """
        content = _read_runbook_md()
        command_tokens = {
            "ingest": "scripts/ingest_games.py",
            "build features": "scripts/build_features.py",
            "train": "scripts/train_models.py",
            "promote": "scripts.promote_models --promote",
            "backtest": "scripts/run_backtest.py",
            "predict": "scripts/generate_current_week_predictions.py",
            "build cache": "scripts/populate_cache.py",
            "serve": "uvicorn api.main:app",
            "rollback": "from models.artifacts import update_manifest",
            "automation": "scripts/friday_pipeline.py",
            "rebuild gold": "scripts/fingerprint_gold.py",
        }
        missing = [
            label for label, token in command_tokens.items() if token not in content
        ]
        assert not missing, f"RUNBOOK.md missing operation commands: {missing}"


class TestRunbookMdCrossReferences:
    """The link-don't-duplicate cross-references (DOC-01/DOC-02) must be present."""

    def test_documents_pipeline_cross_reference(self):
        """RUNBOOK builds on PIPELINE.md (the command source)."""
        content = _read_runbook_md()
        assert "PIPELINE.md" in content

    def test_documents_automation_cross_reference(self):
        """RUNBOOK 'run automation' links AUTOMATION.md, not re-explains it."""
        content = _read_runbook_md()
        assert "AUTOMATION.md" in content

    def test_documents_readme_cross_reference(self):
        """DOC-02 cross-refs README's architecture diagram (no duplicate diagram)."""
        content = _read_runbook_md()
        assert "README.md" in content


class TestRunbookMdArchitectureSection:
    """The DOC-02 Architecture & Data-Flow section anchor must be present."""

    def test_documents_architecture_and_data_flow(self):
        """An Architecture heading + a data-flow reference are present (DOC-02)."""
        content = _read_runbook_md()
        assert "Architecture" in content
        lowered = content.lower()
        assert "data flow" in lowered or "data-flow" in lowered


class TestRunbookMdVerificationBasis:
    """The D-02 per-command verification-basis labels must be present."""

    def test_documents_verification_basis_labels(self):
        """Both the live label and the audit label appear (tiered verification).

        D-02: every command is labeled with how it was verified -- the SAFE
        commands "verified live", the DESTRUCTIVE ones "verified via AUDIT-01
        stage runner; not re-run". Both labels must appear so the tiering is real.
        """
        content = _read_runbook_md()
        lowered = content.lower()
        assert "verified live" in lowered, "missing the live verification-basis label"
        assert "AUDIT-01" in content, "missing the AUDIT-01 verification-basis label"


class TestRunbookMdNoStaleReferences:
    """No Phase-19-deleted file/script may be referenced (no-stale-reference guard)."""

    def test_no_stale_deleted_references(self):
        """The runbook must not reference files/scripts/targets deleted in Phase 19.

        Names every stale reference found so the regression is actionable.
        """
        content = _read_runbook_md()
        forbidden = (
            "deployment/production.env",
            "make snapshot",
            "operational_monitoring.py",
            "validate_models.py",
        )
        present = [token for token in forbidden if token in content]
        assert not present, f"RUNBOOK.md references deleted items: {present}"


class TestRunbookMdBootstrapAndPromote:
    """Phase 25 (D25-16/17): the corrected clean-checkout bootstrap + the Promote/
    Rollback operations must stay in lockstep with the doc.

    Three drift regressions are guarded:

    1. The FALSE "the bundled v1.0 production artifacts ARE present in the checkout"
       claim is GONE (``artifacts/`` is fully gitignored -- a fresh checkout has no
       model dirs and no ``latest.json``; RESEARCH A1 / DIAGNOSIS-NOTES.md).
    2. The corrected setup documents the sanctioned first-manifest bootstrap (a
       one-time ``update_manifest`` per target after the first train) so a clean
       checkout has a written path to its first ``artifacts/latest.json``.
    3. The Promote + Rollback operations name their real surfaces (the ``--promote``
       gate command and the ``update_manifest`` per-key swapper).
    """

    def test_false_bundled_artifacts_claim_absent(self):
        """The stale 'bundled v1.0 artifacts ARE present' claim must be gone (RESEARCH A1)."""
        content = _read_runbook_md()
        assert "production\n  artifacts ARE present in the checkout" not in content, (
            "RUNBOOK.md still carries the false 'bundled artifacts ARE present' claim"
        )
        assert "artifacts ARE present in the checkout" not in content, (
            "RUNBOOK.md still carries the false 'artifacts ARE present in the checkout' claim"
        )

    def test_corrected_bootstrap_path_present(self):
        """The sanctioned first-manifest bootstrap (one-time update_manifest) is documented."""
        content = _read_runbook_md()
        assert "from models.artifacts import update_manifest" in content, (
            "RUNBOOK.md Setup missing the one-time update_manifest bootstrap call"
        )
        assert "update_latest=False" in content, (
            "RUNBOOK.md Setup should explain training defaults update_latest=False "
            "(why a manual first-manifest bootstrap is needed)"
        )
        assert "fully gitignored" in content or "ALL gitignored" in content, (
            "RUNBOOK.md Setup should state artifacts/ is gitignored on a fresh checkout"
        )

    def test_promote_and_rollback_operations_present(self):
        """The Promote (D25-16) + Rollback (D25-17) operations name their real surfaces."""
        content = _read_runbook_md()
        assert "scripts.promote_models --promote" in content, (
            "RUNBOOK.md missing the Promote operation's canonical --promote command"
        )
        assert "Rollback" in content, "RUNBOOK.md missing the Rollback operation"
        assert "ACTIVATION-READOUT.md" in content, (
            "RUNBOOK.md Rollback should reference the pre-swap manifest record in "
            "ACTIVATION-READOUT.md"
        )


class TestRunbookMdPhase30Reconciled:
    """Phase 30 (30-14): the operator record matches the Phase-30 end state.

    Four drift regressions are guarded here:

    1. The stale note describing the feature-build save flag as a no-op with no
       way to disable the gold write is GONE. ``scripts/build_features.py`` now
       defines a real ``--no-save`` (``action="store_false"``, ``dest="save"``),
       so the note described a defect the code no longer has. The absence
       assertion is deliberately narrow -- it targets the specific stale sentence
       fragments, not the word "no-op", so it cannot be satisfied by gutting the
       Build-features section.
    2. The gate's paired baseline is no longer described as the frozen v1.0
       baseline. Since D25-11 it describes the DEPLOYED incumbent, and Phase 30
       re-froze it twice more.
    3. The armed run is documented as the ``--skip-train`` variant, which reuses
       the gate-scored staging dirs rather than re-training and shipping an
       artifact the dry run never scored.
    4. The Phase-30 record is cross-linked, so the Rollback operation's Phase-30
       pre-swap mapping has a written-down source.
    """

    def test_stale_save_flag_noop_note_absent(self):
        """The 'the save flag is a no-op and there is no --no-save' note is gone."""
        content = _read_runbook_md()
        stale_fragments = (
            "there is NO `--no-save`",
            "the\n  flag is effectively always on",
            "flag is effectively always on",
            "there is no `--no-save`)",
        )
        present = [f for f in stale_fragments if f in content]
        assert not present, (
            f"RUNBOOK.md still claims build_features.py has no --no-save switch: {present} "
            "(false -- scripts/build_features.py defines a real --no-save)"
        )

    def test_no_save_documented_as_the_real_off_switch(self):
        """The real read-only-build switch is named in the runbook."""
        content = _read_runbook_md()
        assert "--no-save" in content, (
            "RUNBOOK.md should name --no-save as the real read-only-build switch"
        )

    def test_stale_v1_baseline_claim_absent(self):
        """The gate baseline is no longer called the frozen v1.0 baseline."""
        content = _read_runbook_md()
        assert "frozen v1.0\n  baseline" not in content, (
            "RUNBOOK.md still describes the gate baseline as the frozen v1.0 baseline "
            "(false since D25-11: it describes the DEPLOYED incumbent)"
        )
        assert "frozen v1.0 baseline" not in content, (
            "RUNBOOK.md still describes the gate baseline as the frozen v1.0 baseline "
            "(false since D25-11: it describes the DEPLOYED incumbent)"
        )

    def test_skip_train_armed_variant_documented(self):
        """The reviewed-artifact arming sequence (--skip-train) is documented."""
        content = _read_runbook_md()
        assert "--promote --skip-train" in content, (
            "RUNBOOK.md Promote should document the --skip-train armed variant "
            "(a bare --promote re-trains, shipping an artifact the dry run never scored)"
        )

    def test_cross_links_gated_refit_readout(self):
        """The Phase-30 gated re-fit record is cross-linked."""
        content = _read_runbook_md()
        assert "GATED-REFIT-READOUT.md" in content, (
            "RUNBOOK.md should cross-link GATED-REFIT-READOUT.md (the Phase-30 record)"
        )

    def test_records_the_phase30_refusals_as_retained(self):
        """The two Phase-30 refusals are recorded with their retained incumbents.

        A refusal is a RESULT. The runbook must show what production actually
        points at after a partial pass, not only that a run happened.
        """
        content = _read_runbook_md()
        for version in (
            "wp_20260824_113325",
            "ats_20260605_220128",
            "ou_20260326_163930",
        ):
            assert version in content, (
                f"RUNBOOK.md missing the Phase-30 end-state artifact version: {version}"
            )
        assert "RETAINED" in content, (
            "RUNBOOK.md should record that the two refused targets RETAINED their incumbents"
        )
