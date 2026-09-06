"""Permanent content check for the repo-root AUTOMATION.md.

AUTOMATION.md is the single source of truth for HOW the Friday automation runs
and how to tell whether a run succeeded (AUTO-04). This committed test guards the
required section anchors so the doc cannot silently drift out of sync with the
code it explains:

- the file exists at the repo root,
- the content is ASCII-only (no emoji, per CLAUDE.md),
- the 19-step orchestrator listing is present (18 through plan 31-17 plus the
  plan-31-18 cache step),
- the moved web-cache boundary is described rather than denied,
- the outputs path (``outputs/predictions/``) and log path
  (``logs/friday_pipeline.json``) are documented,
- the single-file-overwrite (no per-run history) note is present,
- the log-only-alerts-by-default note is present,
- the D-04 success-signal triad (``success`` / ``degraded`` / ``failed`` plus the
  matching INFO / WARNING / CRITICAL alert levels) is present, and
- the ``AUDIT-REPORT.md`` + ``PIPELINE.md`` cross-references are present.

This is a permanent committed test, NOT a throwaway script.
"""

from pathlib import Path

# Repo root resolved from this file: tests/unit/test_automation_md.py -> repo root.
REPO_ROOT = Path(__file__).resolve().parents[2]
AUTOMATION_MD = REPO_ROOT / "AUTOMATION.md"


def _read_automation_md() -> str:
    """Read AUTOMATION.md from the repo root."""
    return AUTOMATION_MD.read_text(encoding="utf-8")


class TestAutomationMdExists:
    """AUTOMATION.md must exist at the repo root and be ASCII-only."""

    def test_file_exists_at_repo_root(self):
        """AUTOMATION.md is present at the repo root."""
        assert AUTOMATION_MD.is_file(), f"missing: {AUTOMATION_MD}"

    def test_content_is_ascii(self):
        """Content is ASCII-only (no emoji / non-ASCII, per CLAUDE.md)."""
        content = _read_automation_md()
        assert content.isascii(), "AUTOMATION.md contains non-ASCII characters"


class TestAutomationMdAnchors:
    """The required section anchors must stay present."""

    def test_documents_what_runs_and_when(self):
        """Trigger timing + machine-LOCAL caveat are documented."""
        content = _read_automation_md()
        assert "18:00" in content
        assert "6 PM ET" in content
        assert "machine-LOCAL" in content

    def test_documents_outputs_path(self):
        """The outputs/predictions/ location is documented."""
        content = _read_automation_md()
        assert "outputs/predictions/" in content

    def test_documents_log_path(self):
        """The logs/friday_pipeline.json location is documented."""
        content = _read_automation_md()
        assert "logs/friday_pipeline.json" in content

    def test_documents_single_file_overwrite(self):
        """The single-file-overwrite (no per-run history) note is present."""
        content = _read_automation_md()
        lowered = content.lower()
        assert "single-file overwrite" in lowered
        assert "no per-run history" in lowered

    def test_documents_log_only_alerts_default(self):
        """The log-only-alerts-by-default note is present."""
        content = _read_automation_md()
        assert "log-only" in content.lower()

    def test_documents_email_slack_inert_reality(self):
        """The honest D-05 finding -- email AND Slack are currently inert -- is present.

        Guards against a regression back to the misleading "Slack works as
        documented" claim. AUTOMATION.md must state plainly that both email and
        Slack are inert in the current code (console/log alerts only), and name
        the root cause (AlertManager reads the non-existent
        ``self.settings.monitoring``).
        """
        content = _read_automation_md()
        lowered = content.lower()
        assert "inert" in lowered, "missing the email/Slack-inert honesty note"
        # The precise root cause must be named so the finding stays actionable.
        assert "self.settings.monitoring" in content

    def test_documents_email_slack_enable_keys(self):
        """The email + Slack channel switches are named (for the deferred remedy)."""
        content = _read_automation_md()
        assert "enable_email" in content
        assert "enable_slack" in content

    def test_documents_the_full_step_listing(self):
        """The orchestrator listing/table names EVERY registered step, and the live count.

        Both halves are read from ``pipeline.steps.build_step_registry()`` rather than from a
        hand-maintained literal list. The previous version hardcoded 18 names, so a step added to
        the registry left the document silently short and this guard silently green -- which is
        the doc-drift shape it exists to catch. Now a new registration fails HERE until the
        document describes it.
        """
        from pipeline.steps import build_step_registry

        content = _read_automation_md()
        registry = build_step_registry()

        assert str(len(registry)) in content, (
            f"AUTOMATION.md does not name the live step count ({len(registry)})"
        )
        missing = [step.name for step in registry if step.name not in content]
        assert not missing, f"AUTOMATION.md missing step rows: {missing}"

    def test_documents_the_moved_web_cache_boundary(self):
        """Plan 31-18 -- the orchestrator DOES rebuild the web cache, and the doc says so.

        The stale sentence ("The orchestrator does NOT train, backtest, or rebuild the web
        cache") was TRUE until plan 31-18 registered ``populate_web_cache`` and is false now.
        Guarded in both directions: the denial must be gone, and the replacement must name the
        step, its non-critical registration and its position.
        """
        content = _read_automation_md()
        assert "or rebuild the web cache" not in content, (
            "AUTOMATION.md still denies that the orchestrator rebuilds the web cache; plan 31-18 "
            "registered populate_web_cache as the last step"
        )
        assert "populate_web_cache" in content
        assert "critical=False" in content, (
            "AUTOMATION.md does not record that the cache step is registered non-critical"
        )
        assert "hard-block" in content, (
            "AUTOMATION.md does not record that the /bets hard-block -- not the log-only alert -- "
            "is the real protection against a silently stale bet list"
        )

    def test_documents_d04_success_signal_triad(self):
        """The D-04 triad statuses + matching alert levels are present."""
        content = _read_automation_md()
        # The three log.status values.
        for status in ("success", "degraded", "failed"):
            assert status in content, f"missing status: {status}"
        # The matching alert levels.
        for level in ("INFO", "WARNING", "CRITICAL"):
            assert level in content, f"missing alert level: {level}"

    def test_documents_findings_record(self):
        """A fixed-this-phase findings record + a deferred section are present."""
        content = _read_automation_md()
        assert "## 8. Findings (FIXED this phase)" in content
        assert "Non-correctness findings (deferred)" in content
        # The deferred items named explicitly.
        for item in ("WR-03", "WR-04", "D-11-A"):
            assert item in content, f"missing deferred finding: {item}"

    def test_documents_cross_references(self):
        """AUDIT-REPORT.md + PIPELINE.md cross-references are present."""
        content = _read_automation_md()
        assert "AUDIT-REPORT.md" in content
        assert "PIPELINE.md" in content
