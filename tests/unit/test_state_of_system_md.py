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
STATE_OF_SYSTEM_MD = REPO_ROOT / "STATE-OF-SYSTEM.md"


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
