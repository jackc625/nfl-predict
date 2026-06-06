"""Permanent content check for the repo-root METHODOLOGY.md + the docs/ removal.

METHODOLOGY.md (D-04) is the consolidated, de-staled portfolio deep-dive that
merges the still-true core of the two v1.0-era docs (docs/model_methodology.md +
docs/feature_engineering.md) and CROSS-LINKS current code rather than re-asserting
~1,274 stale lines. This committed test guards:

- the file exists at the repo root,
- the content is ASCII-only (no emoji, per CLAUDE.md / the Windows cp1252
  constraint),
- the current-code cross-links are present (models/, features/, ratings/elo.py),
- the drift landmines are ABSENT: no "placeholder 1500" Elo claim (Elo was
  activated in Phase 11), and no "RandomizedSearchCV" claim (replaced by Optuna
  in Phase 12).

This test file is ALSO the single owner of the docs/ removal assertion (D-01/D-04):
the six stale docs/*.md are deleted and the now-empty docs/ directory is removed.
That check lives in a separate TestStaleDocsRemoved class so it reads independently
of the METHODOLOGY content checks.

This is a permanent committed test, NOT a throwaway script. It intentionally FAILS
until METHODOLOGY.md is written AND the six docs/*.md are deleted (the Wave-2 doc
commit + the deletion turn it GREEN).
"""

from pathlib import Path

# Repo root resolved from this file: tests/unit/test_methodology_md.py -> repo root.
REPO_ROOT = Path(__file__).resolve().parents[2]
METHODOLOGY_MD = REPO_ROOT / "METHODOLOGY.md"


def _read_methodology_md() -> str:
    """Read METHODOLOGY.md from the repo root."""
    return METHODOLOGY_MD.read_text(encoding="utf-8")


class TestMethodologyMdExists:
    """METHODOLOGY.md must exist at the repo root and be ASCII-only."""

    def test_file_exists_at_repo_root(self):
        """METHODOLOGY.md is present at the repo root."""
        assert METHODOLOGY_MD.is_file(), f"missing: {METHODOLOGY_MD}"

    def test_content_is_ascii(self):
        """Content is ASCII-only (no emoji / non-ASCII, per CLAUDE.md)."""
        content = _read_methodology_md()
        assert content.isascii(), "METHODOLOGY.md contains non-ASCII characters"


class TestMethodologyMdCrossLinks:
    """The current-code cross-links (D-04) must be present."""

    def test_cross_links_current_code(self):
        """The methodology cross-links current code rather than re-asserting prose.

        Names the missing cross-links so a drift is actionable.
        """
        content = _read_methodology_md()
        anchors = ("models/", "features/", "ratings/elo.py")
        missing = [anchor for anchor in anchors if anchor not in content]
        assert not missing, f"METHODOLOGY.md missing code cross-links: {missing}"


class TestMethodologyMdNoDriftLandmines:
    """The v1.0-era drift landmines must NOT appear (D-04 still-true filter)."""

    def test_no_placeholder_elo_claim(self):
        """No 'placeholder 1500' Elo claim (Elo activated in Phase 11).

        The v1.0 docs described placeholder 1500.0 Elo; production now runs real
        Elo (6263 per-game snapshots). Any surviving placeholder-Elo prose is a
        false claim. Guard the conjunction so a benign mention of either word
        alone does not trip the test.
        """
        lowered = _read_methodology_md().lower()
        placeholder_elo_claim = "placeholder" in lowered and "1500" in lowered
        assert not placeholder_elo_claim, (
            "METHODOLOGY.md still carries the stale 'placeholder 1500' Elo claim"
        )

    def test_no_randomized_search_cv_claim(self):
        """No 'RandomizedSearchCV' claim (replaced by Optuna in Phase 12)."""
        lowered = _read_methodology_md().lower()
        assert "randomizedsearchcv" not in lowered, (
            "METHODOLOGY.md still references RandomizedSearchCV (replaced by Optuna)"
        )

    def test_no_stale_production_serves_v1_claim(self):
        """Phase 25 (D25-10): the present-tense 'production still serves the v1.0 pre-Elo
        artifacts' claim is gone -- WP + ATS were activated through the gate, O/U retained.
        """
        content = _read_methodology_md()
        assert "production still serves the v1.0 pre-Elo" not in content, (
            "METHODOLOGY.md still claims production serves v1.0 pre-Elo (false post Phase 25)"
        )
        assert "the deployed artifacts are still the v1.0 ones" not in content, (
            "METHODOLOGY.md landmine 3 still claims the deployed artifacts are all v1.0"
        )


class TestMethodologyMdActivationReconciled:
    """Phase 25 (D25-10): METHODOLOGY reflects the post-activation mixed deployed set."""

    def test_cross_links_activation_readout(self):
        """METHODOLOGY cross-links ACTIVATION-READOUT.md for the Phase-25 activation."""
        content = _read_methodology_md()
        assert "ACTIVATION-READOUT.md" in content, (
            "METHODOLOGY.md should cross-link ACTIVATION-READOUT.md (the activation record)"
        )

    def test_records_retained_ou(self):
        """METHODOLOGY records the honest-refusal outcome (O/U retained on v1.0)."""
        content = _read_methodology_md()
        assert "RETAINED" in content, (
            "METHODOLOGY.md should record that O/U retained v1.0 (the honest-refusal outcome)"
        )


class TestStaleDocsRemoved:
    """The six stale docs/*.md and the docs/ directory must be gone (D-01/D-04).

    This class is the SINGLE owner of the docs/ removal assertion across the
    Phase 23 doc guards.
    """

    def test_six_stale_docs_deleted(self):
        """Each of the six stale docs/*.md no longer exists.

        Names every surviving file so the deletion gap is actionable.
        """
        basenames = (
            "operational_runbooks",
            "operational_procedures",
            "GETTING_STARTED",
            "api_documentation",
            "feature_engineering",
            "model_methodology",
        )
        surviving = [
            name for name in basenames if (REPO_ROOT / "docs" / f"{name}.md").exists()
        ]
        assert not surviving, f"stale docs/*.md still present: {surviving}"

    def test_docs_directory_removed(self):
        """The now-empty docs/ directory is removed (D-04)."""
        assert not (REPO_ROOT / "docs").exists(), "docs/ directory still present"
