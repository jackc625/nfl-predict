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
- all 8 common operations are documented (ingest, build features, train,
  backtest, predict, serve, build-cache, run automation),
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
    """All 8 common operations (DOC-01) must be documented."""

    def test_documents_all_eight_operations(self):
        """Each of the 8 operations is documented by its numbered section heading.

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
            "### 4. Backtest",
            "### 5. Predict",
            "### 6. Build cache",
            "### 7. Serve",
            "### 8. Run automation",
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
            "backtest": "scripts/run_backtest.py",
            "predict": "scripts/generate_current_week_predictions.py",
            "build cache": "scripts/populate_cache.py",
            "serve": "uvicorn api.main:app",
            "automation": "scripts/friday_pipeline.py",
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
