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
- section 7 describes the Phase 35 notification paths (no longer "console/log ONLY"): the five
  secrets by env-var name, the TEST and watcher-setup commands, the watcher's 18:30 ET grace and
  season window, the optional watcher test-day line, and that the orchestrator's own log-only
  alert layer is not the notification path,
- the D-04 success-signal triad (``success`` / ``degraded`` / ``failed`` plus the
  matching INFO / WARNING / CRITICAL alert levels) is present, and
- the ``AUDIT-REPORT.md`` + ``PIPELINE.md`` cross-references are present.

This is a permanent committed test, NOT a throwaway script.
"""

from pathlib import Path

# Repo root resolved from this file: tests/unit/test_automation_md.py -> repo root.
REPO_ROOT = Path(__file__).resolve().parents[2]
AUTOMATION_MD = REPO_ROOT / "docs" / "guides" / "AUTOMATION.md"


def _read_automation_md() -> str:
    """Read AUTOMATION.md from the repo root."""
    return AUTOMATION_MD.read_text(encoding="utf-8")


def _section_seven() -> str:
    """Section 7 of AUTOMATION.md: from its ``## 7.`` heading up to the ``## 8.`` heading."""
    content = _read_automation_md()
    return content[content.index("## 7.") : content.index("## 8.")]


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
        """The orchestrator's own alert layer is still described as log-only (D-05).

        Phase 35 kept ``pipeline/alert.py`` as one log line per run outcome; the notification
        path is the separate ``notify/`` package. Section 7 must still say so.
        """
        section = _section_seven()
        assert "pipeline/alert.py" in section
        assert "log-only" in section.lower()

    def test_section_7_describes_the_notification_paths(self):
        """Section 7 tells the truth about the Phase 35 notification paths (SPEC R10).

        It no longer claims alerts are console/log ONLY, names the five secrets by env-var name
        (D-01), the TEST and watcher-setup commands, the watcher's 18:30 ET grace and its
        7-days-before-first-lock season window, and says the old alert layer is not the
        notification path (D-05).
        """
        section = _section_seven()
        heading = section.splitlines()[0]
        assert "console/log ONLY" not in heading, (
            f"section 7 is still log-only: {heading!r}"
        )
        required = (
            "NTFY_TOPIC",
            "RESEND_API_KEY",
            "DIGEST_TO_ADDRESS",
            "HEALTHCHECKS_PING_URL",
            "HEALTHCHECKS_API_KEY",
            "scripts.notify_test",
            "scripts.setup_watcher",
            "18:30",
            "7 days",
            "not the notification path",
        )
        missing = [token for token in required if token not in section]
        assert not missing, f"AUTOMATION.md section 7 is missing: {missing}"

    def test_documents_the_watcher_test_day_line(self):
        """The optional D-16 watcher test-day ``.env`` line is named."""
        assert "WATCHER_TEST_WITHHOLD_DATE" in _section_seven()

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


class TestPhase34LedgerAutomation:
    """Plan 34-21: the forward ledger's scheduled side is documented where the owner looks first.

    The closing wake task, the written Odds API credit budget (LDGR-07), the two deploy keys with
    the rulesets -- including the plain statement that the private backup has none -- and the
    ledger run log each have a section, pinned by its heading.
    """

    def test_documents_every_ledger_section(self):
        """The four Phase-34 sections are present, named in the drift message."""
        content = _read_automation_md()
        headings = (
            "## Closing-line wake task (NFL_Predict_Closing)",
            "## Odds API credit budget",
            "## Ledger pushes and credentials",
            "## Ledger run log",
        )
        missing = [heading for heading in headings if heading not in content]
        assert not missing, f"AUTOMATION.md missing ledger sections: {missing}"

    def test_the_credit_budget_table_keeps_every_month_under_the_quota(self):
        """The published table is present and every month's total and worst case is under 500."""
        content = _read_automation_md()
        rows = [
            [cell.strip() for cell in line.strip("|").split("|")]
            for line in content.splitlines()
            if line.startswith("| 202")
        ]
        assert len(rows) >= 5, (
            f"AUTOMATION.md credit budget table has {len(rows)} months"
        )
        over = [row[0] for row in rows if int(row[5]) >= 500 or int(row[6]) >= 500]
        assert not over, f"AUTOMATION.md budget months at or over 500 credits: {over}"

    def test_states_the_private_backup_has_no_ruleset(self):
        """The backup's only protection is never forcing a push; the doc must not imply more."""
        content = _read_automation_md()
        assert "The private backup repository has NO ruleset." in content, (
            "AUTOMATION.md does not state plainly that the private backup has no ruleset"
        )
