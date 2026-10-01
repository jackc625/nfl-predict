"""Permanent content check for the repo-root PIPELINE.md.

PIPELINE.md is the single source of truth for the canonical run sequence -- 7 stages
when this guard was first written, 8 since Phase 25 inserted the Promote stage.
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

    def test_stage7_lists_the_five_broadcast_pages(self):
        """D-06.2, updated by the Broadcast redesign (2026-10): Stage 7 lists the five nav pages.

        Phase 23.1 added /betting and /season to this list. The Broadcast redesign then merged
        /performance, /backtest, /insights and /betting into /track-record and /how-it-works, so
        the list names the five pages the nav serves; the retired URLs appear only as redirects.
        """
        content = _read_pipeline_md()
        for page in ("/bets", "/season", "/track-record", "/how-it-works"):
            assert f"`{page}`" in content, f"PIPELINE.md Stage 7 missing page: {page}"
        assert "the five top-nav pages" in content, (
            "PIPELINE.md Stage 7 does not say the nav has five pages"
        )
        assert "the seven top-nav pages" not in content, (
            "PIPELINE.md Stage 7 still lists the seven pre-redesign pages"
        )

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


class TestPipelineMdPhase30Reconciled:
    """Phase 30 (30-14): the promote stage reflects the Phase-30 promotion path.

    Three drift regressions are guarded here:

    1. The armed run is documented as the ``--skip-train`` variant, which REUSES
       the gate-scored staging dirs so the artifact that ships is the artifact the
       dry run scored. A bare ``--promote`` re-trains first and ships something
       else; that is the sequence this document must not present as the default.
    2. The stale "frozen v1.0 baseline" description of the gate's paired baseline
       is GONE. Since D25-11 the frozen ``[baseline.*]`` block describes the
       DEPLOYED incumbent, and Phase 30 re-froze it twice more. Calling it the
       "v1.0 baseline" misstates what a candidate is actually measured against.
    3. The Phase-30 readout is cross-linked, so a reader arriving at the promote
       stage can reach the per-target record of the last gated run.
    """

    def test_skip_train_armed_variant_documented(self):
        """The reviewed-artifact arming sequence (--skip-train) is documented."""
        content = _read_pipeline_md()
        assert "--promote --skip-train" in content, (
            "PIPELINE.md Promote stage missing the --skip-train armed variant "
            "(a bare --promote re-trains, shipping an artifact the dry run never scored)"
        )

    def test_stale_v1_baseline_claim_absent(self):
        """The gate's paired baseline is no longer described as the v1.0 baseline."""
        content = _read_pipeline_md()
        assert "frozen v1.0\n  baseline" not in content, (
            "PIPELINE.md still describes the gate baseline as the frozen v1.0 baseline "
            "(false since D25-11: it describes the DEPLOYED incumbent)"
        )
        assert "frozen v1.0 baseline" not in content, (
            "PIPELINE.md still describes the gate baseline as the frozen v1.0 baseline "
            "(false since D25-11: it describes the DEPLOYED incumbent)"
        )

    def test_cross_links_gated_refit_readout(self):
        """The Phase-30 gated re-fit record is cross-linked."""
        content = _read_pipeline_md()
        assert "GATED-REFIT-READOUT.md" in content, (
            "PIPELINE.md should cross-link GATED-REFIT-READOUT.md (the Phase-30 record)"
        )

    def test_no_save_is_documented_as_the_real_off_switch(self):
        """The feature build's real off switch is named (the bare --save cannot turn saving off)."""
        content = _read_pipeline_md()
        assert "--no-save" in content, (
            "PIPELINE.md should name --no-save as the real read-only-build switch "
            "for scripts/build_features.py"
        )


class TestPipelineMdPhase31CacheBoundary:
    """Plan 31-18: stage 7 (build cache) is now ALSO run by the Friday orchestrator.

    PIPELINE.md's weekly-run section previously ended by saying the orchestrator does not
    "rebuild the web cache". That statement is false since ``populate_web_cache`` was registered
    as the last step, and a canonical run sequence that denies a boundary the code crossed is the
    drift this repository has been bitten by more than once. Guarded in both directions.
    """

    def test_the_weekly_run_no_longer_denies_rebuilding_the_web_cache(self):
        """The stale denial is gone from the current-week/weekly-run section."""
        content = _read_pipeline_md()
        assert "or rebuild the web cache" not in content, (
            "PIPELINE.md still claims the Friday orchestrator does not rebuild the web cache "
            "(false since plan 31-18 registered populate_web_cache)"
        )

    def test_the_weekly_run_names_the_cache_step_and_its_registration(self):
        """The replacement names the step, its position and its non-critical registration."""
        content = _read_pipeline_md()
        assert "populate_web_cache" in content, (
            "PIPELINE.md does not name the cache-population step in the weekly run"
        )
        assert "critical=False" in content, (
            "PIPELINE.md does not record that the cache step is registered non-critical, so a "
            "reader cannot tell a degraded run from a failed one"
        )

    def test_the_registry_size_matches_the_live_registry(self):
        """The step count PIPELINE.md quotes is read from the registry, never hardcoded twice."""
        from pipeline.steps import build_step_registry

        content = _read_pipeline_md()
        expected = (
            f"({len(build_step_registry())}-step registry in `pipeline/steps.py`)"
        )
        assert expected in content, (
            f"PIPELINE.md does not quote the live registry size; expected {expected!r}"
        )

    def test_the_style_compile_note_is_present(self):
        """The vendored Tailwind compile has NO make target, and the doc must say so.

        That omission is why compiled-style drift exists in this repo: a template introducing a
        utility class nobody recompiled for renders unstyled, and nothing in the canonical run
        sequence told the author to recompile. Guards the command and the no-make-target fact.
        """
        content = _read_pipeline_md()
        assert "tools/tailwindcss.exe" in content, (
            "PIPELINE.md does not name the vendored Tailwind CLI"
        )
        assert "web/static/css/tailwind-compiled.css" in content, (
            "PIPELINE.md does not name the compiled stylesheet the app serves"
        )
        assert "no `make` target" in content, (
            "PIPELINE.md does not record that the style compile has no make target, which is the "
            "omission that lets compiled-style drift happen unnoticed"
        )
