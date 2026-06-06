"""Permanent content check for the repo-root PIPELINE.md.

PIPELINE.md is the single source of truth for the canonical 7-stage run sequence.
Phase 23.1 corrected two critical errors in this document:

1. Stage 2 "Produces" listed dead silver filenames (``elo_ratings.parquet`` and
   ``team_form.parquet``) that never existed on disk — the same names that caused the
   AUTO-01 blocker (``step_verify_data_artifacts`` gating against non-existent paths).
   The corrected names (``elo_game_snapshots.parquet``, ``team_form_features.parquet``)
   now match the real build-script output and the fixed gate's ``_REQUIRED_ARTIFACTS``.

2. Stage 7 page list omitted ``/betting`` and ``/season``, which Phase 23 added to
   README and RUNBOOK but missed in PIPELINE.md (WR-03 follow-on).

This committed test guards both corrections as drift regressions — the document cannot
silently revert to the broken names or lose the two pages without this test failing.

NOTE: PIPELINE.md intentionally retains a pre-existing non-ASCII em-dash on the Stage 7
line (normalizing it was out of scope per the 23.1-03 SUMMARY deviation).  An
``assert content.isascii()`` assertion WILL FAIL and is therefore deliberately omitted
here.  Unlike ``test_automation_md.py``, this guard does NOT include an ASCII check.

This is a permanent committed test, NOT a throwaway script.
"""

from pathlib import Path

# Repo root resolved from this file: tests/unit/test_pipeline_md.py -> repo root.
REPO_ROOT = Path(__file__).resolve().parents[2]
PIPELINE_MD = REPO_ROOT / "PIPELINE.md"


def _read_pipeline_md() -> str:
    """Read PIPELINE.md from the repo root."""
    return PIPELINE_MD.read_text(encoding="utf-8")


class TestPipelineMdExists:
    """PIPELINE.md must exist at the repo root."""

    def test_file_exists_at_repo_root(self):
        """PIPELINE.md is present at the repo root."""
        assert PIPELINE_MD.is_file(), f"missing: {PIPELINE_MD}"


class TestPipelineMdAnchors:
    """The Phase 23.1 corrections must stay present (drift regression guard)."""

    def test_stage2_corrected_silver_paths_present(self):
        """D-06.1 — Stage 2 lists the two corrected silver filenames.

        build_elo.py writes elo_game_snapshots.parquet and build_team_form.py writes
        team_form_features.parquet.  These are the names ``_REQUIRED_ARTIFACTS`` in
        pipeline/steps.py checks and the names a real build produces.
        """
        content = _read_pipeline_md()
        assert "data/silver/elo_game_snapshots.parquet" in content, (
            "PIPELINE.md Stage 2 missing corrected path: data/silver/elo_game_snapshots.parquet"
        )
        assert "data/silver/team_form_features.parquet" in content, (
            "PIPELINE.md Stage 2 missing corrected path: data/silver/team_form_features.parquet"
        )

    def test_stage2_dead_silver_paths_absent(self):
        """D-06.1 — The two dead silver filenames that caused the AUTO-01 blocker are gone.

        These paths never existed on disk.  If either reappears, the gate will silently
        break again on the next full-mode Friday run.
        """
        content = _read_pipeline_md()
        assert "data/silver/elo_ratings.parquet" not in content, (
            "PIPELINE.md Stage 2 still contains the dead path: data/silver/elo_ratings.parquet"
        )
        assert "data/silver/team_form.parquet" not in content, (
            "PIPELINE.md Stage 2 still contains the dead path: data/silver/team_form.parquet"
        )

    def test_stage7_betting_and_season_pages_present(self):
        """D-06.2 — Stage 7 page list includes /betting and /season.

        These two pages were added in Phase 23 (WR-03); README and RUNBOOK were
        corrected then but PIPELINE.md was missed.  Phase 23.1 closes the gap.
        """
        content = _read_pipeline_md()
        assert "/betting" in content, "PIPELINE.md Stage 7 missing page: /betting"
        assert "/season" in content, "PIPELINE.md Stage 7 missing page: /season"

    def test_stage7_six_page_framing_present(self):
        """D-06.2 — Stage 7 retains the README-authoritative six-page + drill-down framing.

        Guards /games/{id} and /health so the full eight-route set stays documented.
        """
        content = _read_pipeline_md()
        assert "/games/{id}" in content, (
            "PIPELINE.md Stage 7 missing game-detail drill-down: /games/{id}"
        )
        assert "/health" in content, (
            "PIPELINE.md Stage 7 missing health endpoint: /health"
        )


class TestPipelineMdPromoteStage:
    """Phase 25 (D25-16): the Promote stage + the corrected stage count + the stale
    stage-3 latest.json claim removal must stay in lockstep with the doc.

    Phase 25 activated the first gated production swap. Three drift regressions are
    guarded here:

    1. PIPELINE.md now documents **8 stages** (a Promote stage was inserted between
       Train and Backtest); the old "7 stages" count must not reappear.
    2. The Promote stage exists and names its canonical command + the sole per-key
       swapper (``update_manifest``).
    3. The stale stage-3 claim that Train "Produces ... artifacts/latest.json
       manifest" is GONE. Since Plan 24-01, ``save_model_artifact`` defaults
       ``update_latest=False`` and the trainer never overrides it, so training writes
       versioned candidate dirs but never the manifest. Re-introducing the claim that
       training produces ``latest.json`` would be a false claim (new finding #4).
    """

    def test_eight_stage_count_present(self):
        """The corrected 8-stage count is documented and the old 7-stage count is gone."""
        content = _read_pipeline_md()
        assert "8 stages" in content, (
            "PIPELINE.md missing the corrected '8 stages' count"
        )
        assert "8 canonical stages" in content, (
            "PIPELINE.md missing the corrected '8 canonical stages' heading"
        )
        assert "7 stages" not in content, (
            "PIPELINE.md still carries the stale '7 stages' count"
        )
        assert "7 canonical stages" not in content, (
            "PIPELINE.md still carries the stale '7 canonical stages' heading"
        )

    def test_promote_stage_present(self):
        """D25-16 — a Promote stage with its canonical command + sole swapper is present."""
        content = _read_pipeline_md()
        assert "### 4. Promote" in content, (
            "PIPELINE.md missing the '### 4. Promote' stage"
        )
        assert "scripts.promote_models --promote" in content, (
            "PIPELINE.md Promote stage missing the canonical --promote command"
        )
        assert "update_manifest" in content, (
            "PIPELINE.md Promote stage missing the sole per-key swapper update_manifest"
        )

    def test_train_no_longer_claims_it_produces_latest_json(self):
        """new finding #4 — the stale 'stage 3 Train produces latest.json' claim is gone.

        Training does not write the manifest (update_latest=False since Plan 24-01).
        The exact stale phrase that asserted otherwise must not survive.
        """
        content = _read_pipeline_md()
        stale_claim = "model directories plus the\n  `artifacts/latest.json` manifest"
        assert stale_claim not in content, (
            "PIPELINE.md Stage 3 still claims training produces artifacts/latest.json "
            "(false since Plan 24-01: save_model_artifact defaults update_latest=False)"
        )
        assert "update_latest=False" in content, (
            "PIPELINE.md Stage 3 should state training defaults update_latest=False "
            "(it writes candidate dirs only, never the manifest)"
        )
