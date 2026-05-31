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
