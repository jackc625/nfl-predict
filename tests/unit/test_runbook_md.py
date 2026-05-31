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
        """Each of the 8 operations is documented by a substring anchor.

        Names the missing operations so a drift is actionable. Each entry is a
        tuple of acceptable substrings (any one present satisfies the operation),
        because the runbook may name an operation by its script or its verb.
        """
        content = _read_runbook_md()
        operations = {
            "ingest": ("ingest",),
            "build features": ("build_features", "build features"),
            "train": ("train_models", "train models"),
            "backtest": ("run_backtest", "backtest"),
            "predict": ("generate_current_week_predictions", "predict"),
            "serve": ("uvicorn", "serve"),
            "build cache": ("populate_cache", "build-cache", "build cache"),
            "automation": ("AUTOMATION.md",),
        }
        missing = [
            label
            for label, options in operations.items()
            if not any(option in content for option in options)
        ]
        assert not missing, f"RUNBOOK.md missing operations: {missing}"


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
