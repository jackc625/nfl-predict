"""Permanent content guard for the repo-root CLAUDE.md reconciliation.

Phase 23 (Trust & Reproducibility) reconciled the two top-level narrative docs, README.md and
CLAUDE.md, to current reality (D-06), and Phases 25, 30 and 31 extended this guard to each end
state. On 2026-10-04 README.md was rewritten from scratch as a visitor-facing front page and the
owner dropped its history-locking assertions; the pre-rewrite text survives in git history. What
remains guards CLAUDE.md only -- the instruction file every agent session loads:

- it no longer claims "v1.0 MVP shipped 2026-03-28. 10 phases complete" as the current status,
  and references v2.0, v2.1 and v3.0;
- it cross-links ACTIVATION-READOUT.md, GATED-REFIT-READOUT.md and PROFITABILITY-READOUT.md;
- it names the Phase-30 production pointers, including the two the gate REFUSED, and attaches no
  deployment verb to a refused candidate;
- it states that Phase 31 deployed no model, names /bets, reports no target as profitable, and
  presents the CLV-to-ROI divergence as the finding.

This guard does not assert ``content.isascii()``; CLAUDE.md's ASCII rule is asserted by
tests/unit/test_one_lock_rule_source_scan.py.

This is a permanent committed test, NOT a throwaway script.
"""

import re
from pathlib import Path

# Repo root resolved from this file: tests/unit/test_readme_claude_reconciled.py -> repo root.
REPO_ROOT = Path(__file__).resolve().parents[2]
CLAUDE_MD = REPO_ROOT / "CLAUDE.md"


def _read(path: Path) -> str:
    """Read a repo-root markdown file."""
    return path.read_text(encoding="utf-8")


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
    """Phase 30 (30-14): CLAUDE.md describes the Phase-30 end state.

    Phase 30 rebuilt gold, ruled on three feature groups, and re-ran the per-target deploy gate:
    WP PASSED and was promoted; ATS and O/U FAILED and their incumbents were RETAINED. The
    Phase-25 record is kept BESIDE the new one, never overwritten (the supersede-in-place rule),
    so the assertions below are additive.
    """

    def test_claude_cross_links_gated_refit_readout(self):
        """CLAUDE.md cross-links GATED-REFIT-READOUT.md (the Phase-30 record)."""
        content = _read(CLAUDE_MD)
        assert "GATED-REFIT-READOUT.md" in content, (
            "CLAUDE.md should cross-link GATED-REFIT-READOUT.md (the Phase-30 record)"
        )

    def test_claude_names_the_phase30_production_pointers(self):
        """CLAUDE.md names all three Phase-30 serving artifacts, including the two retained.

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
        content = _read(CLAUDE_MD)
        missing = [p for p in pointers if p not in content]
        assert not missing, (
            f"CLAUDE.md missing Phase-30 end-state production pointers: {missing}"
        )

    def test_no_over_claim_word_attached_to_a_refused_target(self):
        """CLAUDE.md does not describe a REFUSED Phase-30 target as deployed.

        Scans each line that names a refused target's artifact version and fails
        if that same line carries a deployment verb. Line-scoped rather than
        document-scoped on purpose: CLAUDE.md legitimately uses "promoted" of WP,
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
        for line in _read(CLAUDE_MD).splitlines():
            lowered = line.lower()
            if not any(v in lowered for v in refused_versions):
                continue
            hits = [w for w in over_claim_words if w in lowered]
            assert not hits, (
                f"CLAUDE.md attaches over-claim word(s) {hits} to a REFUSED "
                f"Phase-30 candidate on this line: {line.strip()!r}"
            )


class TestPhase31Reconciled:
    """Phase 31 (31-19): CLAUDE.md describes the milestone-close end state.

    Additive to the Phase-25 and Phase-30 assertions above. What is guarded here:

    1. CLAUDE.md cross-links ``PROFITABILITY-READOUT.md``.
    2. It names the ``/bets`` page.
    3. It states that Phase 31 DEPLOYED NO MODEL. A phase that shipped a betting page and spent
       the clean split is the easiest phase in the project to misremember as one that changed
       what serves.
    4. It reports no target as profitable. No target came out ``PROFITABLE_CLEAN``.
    """

    def test_claude_cross_links_the_profitability_readout(self):
        """The milestone close is reachable from CLAUDE.md."""
        assert "PROFITABILITY-READOUT.md" in _read(CLAUDE_MD), (
            "CLAUDE.md should cross-link PROFITABILITY-READOUT.md (the Phase-31 milestone close)"
        )

    def test_claude_names_the_bets_page(self):
        """CLAUDE.md describes the web surface, including /bets."""
        assert "/bets" in _read(CLAUDE_MD), "CLAUDE.md does not name the /bets page"

    def test_claude_states_that_phase_31_deployed_no_model(self):
        """The single most misrememberable fact about this phase."""
        content = _read(CLAUDE_MD).lower()
        assert "deployed no model" in content, (
            "CLAUDE.md does not state that Phase 31 deployed no model."
        )
        assert "byte-unchanged" in content, (
            "CLAUDE.md does not state that the production swap surface is byte-unchanged by "
            "Phase 31"
        )

    def test_claude_reports_no_profitable_target(self):
        """No target cleared its pre-registered ROI test, and CLAUDE.md must not round up.

        Checked over whitespace-FLATTENED text and inside a preceding window, not line by line:
        the document is hard-wrapped, so the negation and the token it negates can land on
        different lines.
        """
        window = 80
        flat = re.sub(r"\s+", " ", _read(CLAUDE_MD))
        for match in re.finditer(r"(?<!UN)PROFITABLE_CLEAN", flat):
            before = flat[max(0, match.start() - window) : match.start()].lower()
            assert "no target" in before or "never called profitable" in before, (
                f"CLAUDE.md uses PROFITABLE_CLEAN without a negation in the preceding "
                f"{window} characters: ...{flat[max(0, match.start() - window) : match.end()]!r}. "
                "No target came out profitable on the clean 2025 split."
            )

    def test_claude_states_the_clv_to_roi_divergence_as_the_finding(self):
        """The headline is the divergence, not the return."""
        content = _read(CLAUDE_MD).lower()
        assert "closing-line value is not profitability" in content or (
            "clv-to-roi divergence" in content
        ), "CLAUDE.md does not present the CLV-to-ROI divergence as the finding."
