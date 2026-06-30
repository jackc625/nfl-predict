"""Permanent doc-drift guard for the repo-root LINE-MOVEMENT-READOUT.md (Phase 29, SIG-04 / D-16).

Mirrors ``tests/unit/test_signal_lift_readout_md.py`` (the committed doc-drift-guard pattern): a
PERMANENT committed test, NOT a throwaway ``scripts/check_*.py``. It guards the SIG-04 / D-16
deliverable ``LINE-MOVEMENT-READOUT.md`` -- the budget-gate go/no-go record -- against the
failure modes that would silently corrupt the gate record:

  - the file is missing or not at the repo root,
  - the content carries non-ASCII (emoji / cp1252-hostile) characters (CLAUDE.md hard constraint),
  - a required section (the tiered cost, the plausibility argument, the decision, the lift
    placeholder, the scope-down/slip note) was silently dropped,
  - the SCREEN-NOT-DEPLOY invariant was violated -- the doc must say "carried to Phase 30" and
    must NEVER say "deployed" / "proven" (a budget-gate spike is not a deploy decision, D-16),
  - the machine-readable ``selected_branch:`` marker is missing, duplicated, or carries an
    unknown token (downstream branch gating reads this literal token, not prose -- review 29-01
    LOW: assert EXACTLY ONE valid branch token).

This is the Plan-29-01 language-invariant guard; Plan 29-07 (backfill path only) extends it with
the numeric lift-reproduction check, exactly as the Phase-28 readout guard does.

ASCII only, no emoji (CLAUDE.md).
"""

from __future__ import annotations

import re
from pathlib import Path

# Repo root resolved from this file: tests/unit/test_line_movement_readout_md.py -> repo root.
REPO_ROOT = Path(__file__).resolve().parents[2]
READOUT_MD = REPO_ROOT / "LINE-MOVEMENT-READOUT.md"

# Required section markers (the D-16 five-section contract). A substring of each section header so
# a benign reword does not trip the guard but a dropped section does.
_REQUIRED_SECTION_MARKERS = (
    "Tiered cost",  # 1 -- the (a)/(b)/(c) x window credit+dollar menu
    "Plausibility (D-02)",  # 2 -- the non-redundancy argument
    "Decision:",  # 3 -- the owner branch decision
    "Lift results",  # 4 -- the backfill-path lift placeholder
    "Honest scope-down",  # 5 -- the scope-down / slip note
)

# The cost-method honesty stamp (D-04: estimate, free key not called) + the freshness re-confirm
# date (review 29-01 LOW) must both be present in the tiered-cost section.
_REQUIRED_PHRASES = (
    "carried to Phase 30",  # the screen-not-deploy carry phrase
    "checked 2026-06-29",  # the 7-day-freshness re-confirmation stamp (D-04)
)

# The screen-not-deploy invariant: these over-claim words must NEVER appear (mirrors the negative
# grep in the plan's acceptance: ``grep -ci 'deployed\\|proven'`` must return 0).
_FORBIDDEN_WORDS = ("deployed", "proven")

# The machine-readable branch marker contract (review 29-01 LOW): exactly one line of the form
# ``selected_branch: <token>`` with a token from this set.
_VALID_BRANCH_TOKENS = frozenset(
    {"PENDING", "full-backfill", "forward-collect-only", "slip"}
)
_BRANCH_MARKER_RE = re.compile(r"^selected_branch:\s*(\S+)\s*$", re.MULTILINE)


def _read_readout() -> str:
    """Read LINE-MOVEMENT-READOUT.md from the repo root."""
    return READOUT_MD.read_text(encoding="utf-8")


class TestReadoutExists:
    """LINE-MOVEMENT-READOUT.md must exist at the repo root and be ASCII-only."""

    def test_file_exists_at_repo_root(self) -> None:
        """The SIG-04 / D-16 deliverable is present at the repo root (NOT under .planning/)."""
        assert READOUT_MD.is_file(), f"missing: {READOUT_MD}"

    def test_content_is_ascii(self) -> None:
        """Content is ASCII-only (no emoji / non-ASCII, per CLAUDE.md)."""
        content = _read_readout()
        assert content.isascii(), (
            "LINE-MOVEMENT-READOUT.md contains non-ASCII characters"
        )


class TestReadoutSections:
    """All five required sections (D-16) must be present."""

    def test_all_required_section_markers_present(self) -> None:
        """Every required section marker is present; names any missing so a drop is actionable."""
        content = _read_readout()
        missing = [m for m in _REQUIRED_SECTION_MARKERS if m not in content]
        assert not missing, (
            f"LINE-MOVEMENT-READOUT.md missing required sections: {missing}"
        )

    def test_tiered_cost_table_has_all_three_rungs(self) -> None:
        """Section 1 prices the full (a)/(b)/(c) tiered menu (D-03)."""
        content = _read_readout()
        for rung in ("(a) full trajectory", "(b) full trajectory", "(c) two-anchor"):
            assert rung in content, f"the tiered cost menu is missing rung {rung!r}"

    def test_cost_is_an_estimate_not_a_call(self) -> None:
        """Section 1 is headed as an ESTIMATE from published pricing (D-04 -- endpoint not called)."""
        content = _read_readout()
        assert "estimated from" in content.lower(), (
            "the cost section must state it is ESTIMATED from published pricing (D-04)"
        )
        assert "not called" in content.lower(), (
            "the cost section must state the historical endpoint was NOT called (D-04)"
        )


class TestScreenNotDeployInvariant:
    """The doc must say 'carried to Phase 30' and must NEVER say 'deployed' / 'proven' (D-16)."""

    def test_required_phrases_present(self) -> None:
        """The carry phrase + the freshness re-confirmation stamp are present."""
        content = _read_readout()
        missing = [p for p in _REQUIRED_PHRASES if p not in content]
        assert not missing, (
            f"LINE-MOVEMENT-READOUT.md missing required phrases: {missing}"
        )

    def test_no_over_claim_words(self) -> None:
        """The over-claim words 'deployed' / 'proven' must NOT appear (a spike is not a deploy)."""
        lowered = _read_readout().lower()
        present = [w for w in _FORBIDDEN_WORDS if w in lowered]
        assert not present, (
            f"LINE-MOVEMENT-READOUT.md over-claims a budget-gate as a deploy: {present} "
            "(D-16 -- use 'carried to Phase 30', never 'deployed' / 'proven')"
        )

    def test_no_literal_api_key(self) -> None:
        """The readout records dollars/credits only; the ODDS_API_KEY is never echoed."""
        lowered = _read_readout().lower()
        assert "apikey=" not in lowered, (
            "the readout must not echo a literal apiKey= value"
        )


class TestSelectedBranchMarker:
    """Exactly one machine-readable ``selected_branch:`` token with a valid value (review 29-01 LOW)."""

    def test_exactly_one_marker_line(self) -> None:
        """There is EXACTLY ONE ``selected_branch:`` marker line (not zero, not multiple)."""
        matches = _BRANCH_MARKER_RE.findall(_read_readout())
        assert len(matches) == 1, (
            f"expected exactly one 'selected_branch:' marker line, found {len(matches)}: {matches}"
        )

    def test_marker_token_is_valid(self) -> None:
        """The marker's token is one of {PENDING, full-backfill, forward-collect-only, slip}."""
        matches = _BRANCH_MARKER_RE.findall(_read_readout())
        assert len(matches) == 1, (
            f"expected exactly one 'selected_branch:' marker line, found {len(matches)}: {matches}"
        )
        token = matches[0]
        assert token in _VALID_BRANCH_TOKENS, (
            f"selected_branch token {token!r} is not one of {sorted(_VALID_BRANCH_TOKENS)}"
        )
