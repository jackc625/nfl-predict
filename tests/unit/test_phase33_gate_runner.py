"""Stage one writes nothing, stage two cannot run without it, and the copy precedes the swap.

Phase 33, Plan 33-08 Task 3(d) (COLD-04, D33-13, T-33-38 / T-33-39 / T-33-41d / T-33-41e).

WHERE PLAN 33-08 LEFT THE CHOICE, AND WHAT WAS CHOSEN
--------------------------------------------------------
Task 3(d) says the stage-1-writes-nothing and stage-2-requires-stage-1 tests may live in
``test_phase33_preregistration.py``'s sibling coverage OR in the runner's own module
test, "whichever the executor chooses". They live HERE, in the runner's own module test,
so ``test_phase33_preregistration.py`` stays a pure pre-registration module that Plan
33-16 can EXTEND by appending its six edge thresholds rather than by editing around
unrelated coverage.

WHY THE WRITE-NOTHING PROOF BRACKETS A DIGEST AND NOT A GIT STATUS
--------------------------------------------------------------------
Phases 28-31 reported "the data boundary held" using ``git status --porcelain``, and
``.gitignore`` blankets those paths, so that command was STRUCTURALLY INCAPABLE OF
FAILING. The instrument here is ``tests.data_boundary.digest_tree`` over a SANDBOXED
production root, taken before and after a full stage-one run and compared key by key.

WHY THE SANDBOX IS NOT OPTIONAL
--------------------------------
Plan 33-06 overwrote all three production gold matrices because a test called a helper
that wrote, without the sandbox fixture; the guard caught it at TEARDOWN, after the
write. Every root in this module comes from ``tests.phase33_gate_fixtures.sandbox_roots``,
which REFUSES -- before doing anything -- to hand back a path inside the repository's real
``artifacts/`` tree. A stop beats a detector.

WHY COPY-BEFORE-SWAP IS TESTED AS A CRASH AND NOT AS AN ORDERING COMMENT
--------------------------------------------------------------------------
``models.artifacts.update_manifest`` rewrites only the per-target POINTER. Aiming
production ``latest.json`` at a version whose files exist only under staging leaves
``load_model_artifact`` raising ``FileNotFoundError`` -- a production outage from a
successful-looking promotion. The recovery property is asserted directly: the manifest
writer is made to raise AFTER the copies, and the OLD manifest is then shown to be
present, parseable, and pointing only at versions that resolve on disk.

Run this module:  uv run pytest tests/unit/test_phase33_gate_runner.py -q

ASCII only, no emoji (CLAUDE.md hard constraint).
"""

from __future__ import annotations

import ast
import inspect
import json
from pathlib import Path

import pytest

from scripts import run_phase33_gate as runner
from tests import phase33_gate_fixtures as fx
from tests.data_boundary import digest_tree


def _passing_scoring(target: str, ids: list[str]) -> runner.TargetScoring:
    metrics = (
        {"accuracy": 1.0, "ece": 0.0, "brier_score": 0.0}
        if target == "wp"
        else {"mae": 0.0}
    )
    value = 0.5 if target == "wp" else 4.0
    return runner.TargetScoring(
        target=target,
        candidate_version=fx.CANDIDATE_VERSIONS[target],
        incumbent_version=fx.INCUMBENT_VERSIONS[target],
        candidate_bundle=fx.candidate_bundle(target, **metrics),
        incumbent_scored=fx.scored_frame(target, ids, value=value),
        candidate_scored=fx.scored_frame(target, ids, value=value),
        gold=fx.gold_frame(target, ids),
        paired_delta=fx.paired_delta_frame(ids, delta=0.4),
    )


def _run_stage_one(tmp_path: Path, roots: fx.SandboxRoots, targets=("wp", "ats", "ou")):
    ids = fx.game_ids(120)
    scorings = {t: _passing_scoring(t, ids) for t in targets}
    return runner.stage_one_judge(
        scorer=lambda target, _s, _a: scorings[target],
        cfg=fx.gate_cfg(),
        targets=targets,
        artifacts_dir=roots.artifacts,
        staging_dir=roots.staging,
        verdict_record_path=tmp_path / "outputs" / "verdict.json",
        committed_verdict_path=tmp_path / "config" / "verdict.toml",
        min_free_bytes=0,
        now="2026-09-12T00:00:00+00:00",
    )


# ---------------------------------------------------------------------------
# The pre-flight
# ---------------------------------------------------------------------------


class TestThePreflightFailsTheEnvironmentBeforeAnythingIsMeasured:
    """Four failure modes, driven separately, each naming its own check."""

    def test_a_healthy_sandbox_passes_and_reports_what_it_checked(
        self, tmp_path: Path
    ) -> None:
        roots = fx.sandbox_roots(tmp_path)
        report = runner.preflight_health_check(
            artifacts_dir=roots.artifacts, staging_dir=roots.staging, min_free_bytes=0
        )
        assert set(report) == {
            "staging_writable",
            "staging_clear",
            "artifacts_writable",
            "incumbents_resolve",
            "free_disk",
        }
        assert report["incumbents_resolve"]["wp"] == fx.INCUMBENT_VERSIONS["wp"]

    def test_an_unwritable_staging_root_fails_by_name(self, tmp_path: Path) -> None:
        """Driven by pointing staging at a regular FILE -- portable, unlike chmod."""
        roots = fx.sandbox_roots(tmp_path)
        blocked = tmp_path / "staging_is_a_file"
        blocked.write_text("not a directory", encoding="utf-8")
        with pytest.raises(runner.PreflightFailedError) as excinfo:
            runner.preflight_health_check(
                artifacts_dir=roots.artifacts, staging_dir=blocked, min_free_bytes=0
            )
        assert "staging_writable" in str(excinfo.value)

    def test_a_staging_root_that_is_not_clear_fails_by_name(
        self, tmp_path: Path
    ) -> None:
        """A stale candidate dir can be selected by newest-by-name and scored."""
        roots = fx.sandbox_roots(tmp_path)
        (roots.staging / "wp_19990101_000000").mkdir(parents=True)
        with pytest.raises(runner.PreflightFailedError) as excinfo:
            runner.preflight_health_check(
                artifacts_dir=roots.artifacts,
                staging_dir=roots.staging,
                min_free_bytes=0,
            )
        assert "staging_clear" in str(excinfo.value)
        assert "wp_19990101_000000" in str(excinfo.value)

    def test_an_absent_incumbent_artifact_fails_by_name(self, tmp_path: Path) -> None:
        """The incumbent is both the paired comparator and the rollback target."""
        roots = fx.sandbox_roots(tmp_path)
        (roots.artifacts / fx.INCUMBENT_VERSIONS["ats"] / "metadata.json").unlink()
        with pytest.raises(runner.PreflightFailedError) as excinfo:
            runner.preflight_health_check(
                artifacts_dir=roots.artifacts,
                staging_dir=roots.staging,
                min_free_bytes=0,
            )
        assert "incumbents_resolve" in str(excinfo.value)
        assert "ats" in str(excinfo.value)

    def test_free_disk_below_the_declared_floor_fails_by_name(
        self, tmp_path: Path
    ) -> None:
        roots = fx.sandbox_roots(tmp_path)
        with pytest.raises(runner.PreflightFailedError) as excinfo:
            runner.preflight_health_check(
                artifacts_dir=roots.artifacts,
                staging_dir=roots.staging,
                min_free_bytes=1 << 60,
            )
        assert "free_disk" in str(excinfo.value)

    def test_a_missing_production_manifest_fails_by_name(self, tmp_path: Path) -> None:
        roots = fx.sandbox_roots(tmp_path)
        (roots.artifacts / "latest.json").unlink()
        with pytest.raises(runner.PreflightFailedError) as excinfo:
            runner.preflight_health_check(
                artifacts_dir=roots.artifacts,
                staging_dir=roots.staging,
                min_free_bytes=0,
            )
        assert "incumbents_resolve" in str(excinfo.value)

    def test_the_preflight_runs_BEFORE_any_scoring_call(self, tmp_path: Path) -> None:
        """A scorer stub that records whether it was invoked. It must not have been."""
        roots = fx.sandbox_roots(tmp_path)
        (roots.staging / "ou_19990101_000000").mkdir(parents=True)
        invoked: list[str] = []

        def _recording_scorer(target: str, _s: Path, _a: Path) -> runner.TargetScoring:
            invoked.append(target)
            raise AssertionError("the scorer must not run when the pre-flight fails")

        with pytest.raises(runner.PreflightFailedError):
            runner.stage_one_judge(
                scorer=_recording_scorer,
                cfg=fx.gate_cfg(),
                artifacts_dir=roots.artifacts,
                staging_dir=roots.staging,
                verdict_record_path=tmp_path / "outputs" / "verdict.json",
                committed_verdict_path=tmp_path / "config" / "verdict.toml",
                min_free_bytes=0,
            )
        assert invoked == []

    def test_a_shared_staging_and_production_root_is_refused(
        self, tmp_path: Path
    ) -> None:
        """'Staged' means nothing if staging IS production."""
        roots = fx.sandbox_roots(tmp_path)
        with pytest.raises(runner.PreflightFailedError) as excinfo:
            runner.stage_one_judge(
                scorer=lambda *_a: None,
                cfg=fx.gate_cfg(),
                artifacts_dir=roots.artifacts,
                staging_dir=roots.artifacts,
                verdict_record_path=tmp_path / "outputs" / "verdict.json",
                committed_verdict_path=tmp_path / "config" / "verdict.toml",
                min_free_bytes=0,
            )
        assert "staging_is_separate" in str(excinfo.value)


# ---------------------------------------------------------------------------
# Stage one writes nothing into production
# ---------------------------------------------------------------------------


class TestStageOneLeavesProductionByteUnchanged:
    """T-33-39, proven by a digest bracket over the WHOLE sandboxed production tree."""

    def test_the_production_tree_digest_is_identical_before_and_after(
        self, tmp_path: Path
    ) -> None:
        roots = fx.sandbox_roots(tmp_path)
        fx.stage_candidate_dirs(roots, ("wp", "ats", "ou"))
        before = digest_tree(roots.artifacts)
        assert before, "the bracket must have something to compare, or it is vacuous"
        _run_stage_one(tmp_path, roots)
        assert digest_tree(roots.artifacts) == before

    def test_the_manifest_specifically_is_unchanged(self, tmp_path: Path) -> None:
        roots = fx.sandbox_roots(tmp_path)
        _run_stage_one(tmp_path, roots)
        manifest = json.loads((roots.artifacts / "latest.json").read_text("utf-8"))
        assert manifest == fx.INCUMBENT_VERSIONS

    def test_the_bracket_would_notice_a_write_so_it_is_not_vacuous(
        self, tmp_path: Path
    ) -> None:
        """Control: the same instrument, with a deliberate manifest write in between."""
        roots = fx.sandbox_roots(tmp_path)
        before = digest_tree(roots.artifacts)
        (roots.artifacts / "latest.json").write_text(
            '{"wp": "moved"}', encoding="utf-8"
        )
        assert digest_tree(roots.artifacts) != before

    def test_stage_one_makes_no_manifest_write_call_at_all(self) -> None:
        """AST backstop for the runtime bracket above."""
        tree = ast.parse(inspect.getsource(runner))
        fn = next(
            node
            for node in ast.walk(tree)
            if isinstance(node, ast.FunctionDef) and node.name == "stage_one_judge"
        )
        attrs = {n.attr for n in ast.walk(fn) if isinstance(n, ast.Attribute)}
        assert attrs.isdisjoint(
            {"update_manifest", "write_text", "to_json", "replace"}
        ), f"stage_one_judge reaches a write attribute: {sorted(attrs)}"

    def test_stage_one_still_writes_its_own_verdict_record(
        self, tmp_path: Path
    ) -> None:
        """Writing nothing into PRODUCTION is not writing nothing at all."""
        roots = fx.sandbox_roots(tmp_path)
        _run_stage_one(tmp_path, roots)
        assert (tmp_path / "outputs" / "verdict.json").exists()
        assert (tmp_path / "config" / "verdict.toml").exists()

    def test_the_committed_half_carries_the_do_not_hand_edit_header(
        self, tmp_path: Path
    ) -> None:
        roots = fx.sandbox_roots(tmp_path)
        _run_stage_one(tmp_path, roots)
        text = (tmp_path / "config" / "verdict.toml").read_text("utf-8")
        assert "GENERATOR OUTPUT" in text
        assert "Do NOT hand-edit" in text
        assert "fix_cycle_allowance = 0" in text

    def test_every_verdict_row_carries_the_judge_version_and_a_64_hex_digest(
        self, tmp_path: Path
    ) -> None:
        roots = fx.sandbox_roots(tmp_path)
        record = _run_stage_one(tmp_path, roots)
        for row in record.targets.values():
            assert row["judge_version"] == runner.deploy_gate.JUDGE_VERSION
            assert len(row["judge_code_digest"]) == 64

    def test_the_record_declares_the_five_secondary_scalars(
        self, tmp_path: Path
    ) -> None:
        roots = fx.sandbox_roots(tmp_path)
        record = _run_stage_one(tmp_path, roots)
        assert len(record.secondary_scalar_names) == 5


class TestTheAllowanceIsEnforcedAtTheRunner:
    """A second candidate for a judged target is refused, not quietly re-judged."""

    def test_a_second_stage_one_for_a_judged_target_raises(
        self, tmp_path: Path
    ) -> None:
        roots = fx.sandbox_roots(tmp_path)
        _run_stage_one(tmp_path, roots)
        fresh = fx.sandbox_roots(tmp_path / "second")
        with pytest.raises(runner.FixCycleAllowanceExceededError) as excinfo:
            runner.stage_one_judge(
                scorer=lambda target, _s, _a: _passing_scoring(
                    target, fx.game_ids(120)
                ),
                cfg=fx.gate_cfg(),
                artifacts_dir=fresh.artifacts,
                staging_dir=fresh.staging,
                verdict_record_path=tmp_path / "outputs" / "verdict.json",
                committed_verdict_path=tmp_path / "second.toml",
                min_free_bytes=0,
            )
        assert "fix-cycle allowance is 0" in str(excinfo.value)


# ---------------------------------------------------------------------------
# Stage two
# ---------------------------------------------------------------------------


class TestStageTwoCannotRunWithoutStageOne:
    """T-33-38: the property is structural, not procedural."""

    def test_it_raises_missing_stage_one_verdict_with_no_record(
        self, tmp_path: Path
    ) -> None:
        roots = fx.sandbox_roots(tmp_path)
        with pytest.raises(runner.MissingStageOneVerdictError) as excinfo:
            runner.stage_two_promote(
                verdict_record_path=tmp_path / "outputs" / "nothing.json",
                artifacts_dir=roots.artifacts,
                staging_dir=roots.staging,
            )
        assert "no stage-one verdict record" in str(excinfo.value)

    def test_it_rejects_a_malformed_record_rather_than_promoting_from_it(
        self, tmp_path: Path
    ) -> None:
        roots = fx.sandbox_roots(tmp_path)
        record_path = tmp_path / "outputs" / "verdict.json"
        record_path.parent.mkdir(parents=True, exist_ok=True)
        record_path.write_text(
            json.dumps(
                {
                    "judge_version": "x",
                    "judge_code_digest": "y",
                    "rendered_at": "z",
                    "fix_cycle_allowance": 0,
                    "verdict_states": list(runner.VERDICT_STATES),
                    "secondary_scalar_names": [],
                    "targets": {
                        "wp": {
                            "verdict": "PASS",
                            "paired_statistic": None,
                            "p_value": None,
                            "candidate_version": "wp_x",
                        }
                    },
                }
            ),
            encoding="utf-8",
        )
        with pytest.raises(runner.MalformedVerdictRecordError):
            runner.stage_two_promote(
                verdict_record_path=record_path,
                artifacts_dir=roots.artifacts,
                staging_dir=roots.staging,
            )


class TestStageTwoCopiesBeforeItSwaps:
    """T-33-41d: a pointer at a staging-only version is a production outage."""

    def _staged_pass(self, tmp_path: Path) -> tuple[fx.SandboxRoots, Path]:
        roots = fx.sandbox_roots(tmp_path)
        fx.stage_candidate_dirs(roots, ("wp", "ats", "ou"))
        _run_stage_one(tmp_path, roots)
        return roots, tmp_path / "outputs" / "verdict.json"

    def test_a_promotion_leaves_every_manifest_pointer_resolvable(
        self, tmp_path: Path
    ) -> None:
        roots, record_path = self._staged_pass(tmp_path)
        promoted = runner.stage_two_promote(
            verdict_record_path=record_path,
            artifacts_dir=roots.artifacts,
            staging_dir=roots.staging,
        )
        assert promoted == ("wp", "ats", "ou")
        manifest = json.loads((roots.artifacts / "latest.json").read_text("utf-8"))
        for version in manifest.values():
            assert (roots.artifacts / version).is_dir(), version

    def test_the_blend_pointer_survives_untouched(self, tmp_path: Path) -> None:
        """T-33-42: four entries, not three independent slots."""
        roots, record_path = self._staged_pass(tmp_path)
        runner.stage_two_promote(
            verdict_record_path=record_path,
            artifacts_dir=roots.artifacts,
            staging_dir=roots.staging,
        )
        manifest = json.loads((roots.artifacts / "latest.json").read_text("utf-8"))
        assert manifest["blend"] == fx.INCUMBENT_VERSIONS["blend"]

    def test_a_non_promoted_target_keeps_its_incumbent_pointer(
        self, tmp_path: Path
    ) -> None:
        roots = fx.sandbox_roots(tmp_path)
        fx.stage_candidate_dirs(roots, ("wp",))
        _run_stage_one(tmp_path, roots, targets=("wp",))
        runner.stage_two_promote(
            verdict_record_path=tmp_path / "outputs" / "verdict.json",
            artifacts_dir=roots.artifacts,
            staging_dir=roots.staging,
        )
        manifest = json.loads((roots.artifacts / "latest.json").read_text("utf-8"))
        assert manifest["wp"] == fx.CANDIDATE_VERSIONS["wp"]
        assert manifest["ats"] == fx.INCUMBENT_VERSIONS["ats"]
        assert manifest["ou"] == fx.INCUMBENT_VERSIONS["ou"]

    def test_a_crash_between_the_copies_and_the_swap_leaves_the_old_manifest_usable(
        self, tmp_path: Path, monkeypatch
    ) -> None:
        """THE recovery property, and the reason the copy runs first."""
        roots, record_path = self._staged_pass(tmp_path)

        def _explode(*_args, **_kwargs) -> None:
            raise RuntimeError("simulated crash after the artifact copies")

        monkeypatch.setattr(runner, "update_manifest", _explode)
        with pytest.raises(RuntimeError, match="simulated crash"):
            runner.stage_two_promote(
                verdict_record_path=record_path,
                artifacts_dir=roots.artifacts,
                staging_dir=roots.staging,
            )

        manifest_path = roots.artifacts / "latest.json"
        assert manifest_path.exists(), "the old manifest must still be present"
        manifest = json.loads(manifest_path.read_text("utf-8"))
        assert manifest == fx.INCUMBENT_VERSIONS, "no pointer may have moved"
        for version in manifest.values():
            assert (roots.artifacts / version).is_dir(), version

    def test_the_copies_really_did_land_before_the_crash(
        self, tmp_path: Path, monkeypatch
    ) -> None:
        """Otherwise the recovery test above would pass on a stage two that did nothing."""
        roots, record_path = self._staged_pass(tmp_path)
        monkeypatch.setattr(
            runner,
            "update_manifest",
            lambda *_a, **_k: (_ for _ in ()).throw(RuntimeError("x")),
        )
        with pytest.raises(RuntimeError):
            runner.stage_two_promote(
                verdict_record_path=record_path,
                artifacts_dir=roots.artifacts,
                staging_dir=roots.staging,
            )
        for target in ("wp", "ats", "ou"):
            assert (roots.artifacts / fx.CANDIDATE_VERSIONS[target]).is_dir()

    def test_the_copy_call_precedes_the_manifest_call_in_source(self) -> None:
        """Structural backstop for the crash test."""
        src = inspect.getsource(runner.stage_two_promote)
        assert "_promote_artifact_dir" in src
        assert "update_manifest" in src
        assert src.find("_promote_artifact_dir") < src.find("update_manifest")

    def test_stage_two_never_rewrites_the_whole_manifest_itself(self) -> None:
        """Pitfall 4, named in update_manifest's own docstring."""
        src = inspect.getsource(runner.stage_two_promote)
        assert "json.dump" not in src
        assert "write_text" not in src

    def test_stage_two_recomputes_no_verdict(self) -> None:
        src = inspect.getsource(runner.stage_two_promote)
        assert "evaluate_target" not in src
        assert "render_target_verdict" not in src


class TestTheCliIsStageOneByDefault:
    """Matching promote_models' dry-run-by-default posture."""

    def test_promote_defaults_to_false(self) -> None:
        assert runner.parse_args([]).promote is False

    def test_promote_is_behind_an_explicit_flag(self) -> None:
        assert runner.parse_args(["--promote"]).promote is True

    def test_a_bare_invocation_only_runs_the_preflight(
        self, tmp_path: Path, capsys
    ) -> None:
        roots = fx.sandbox_roots(tmp_path)
        exit_code = runner.main(
            [
                "--artifacts-dir",
                str(roots.artifacts),
                "--staging-dir",
                str(roots.staging),
            ]
        )
        assert exit_code == 0
        assert "PRE-FLIGHT PASSED" in capsys.readouterr().out
        assert json.loads((roots.artifacts / "latest.json").read_text("utf-8")) == (
            fx.INCUMBENT_VERSIONS
        )


class TestTheSandboxHelperIsFailClosed:
    """The helper every test here depends on refuses the real production root."""

    def test_it_refuses_a_root_inside_the_real_artifacts_tree(self) -> None:
        with pytest.raises(fx.SandboxEscapeError):
            fx.sandbox_roots(fx.REAL_ARTIFACTS_ROOT / "phase33_sandbox_probe")

    def test_the_real_artifacts_tree_is_untouched_by_that_refusal(self) -> None:
        assert not (fx.REAL_ARTIFACTS_ROOT / "phase33_sandbox_probe").exists()
