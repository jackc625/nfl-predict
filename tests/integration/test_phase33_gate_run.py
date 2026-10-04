"""Stage two: one owner-authorised promotion, copy before swap, against SANDBOXED roots.

Phase 33, Plan 33-15 Task 4 (COLD-01 / COLD-04, D33-13).

WHAT THIS MODULE PROVES, AND WHERE IT PROVES IT
-----------------------------------------------
Every behavioural assertion here runs against a sandboxed PAIR of artifact roots built by
``tests.phase33_gate_fixtures.sandbox_roots``, which is fail-closed: it REFUSES to hand
back a root resolving inside the repository's real ``artifacts/`` tree. The three
assertions that read committed RECORDS -- the verdict TOML, the phase-33 state slots and
the live manifest -- are reading, never writing.

WHY STAGE TWO NEEDED AN OVERRIDE AT ALL, which is the part a later reader will want.
``stage_two_promote`` shipped in Plan 33-08 promoting exactly ``record.passing_targets()``
and nothing else, and that was right for the phase it was written in. The owner's standing
ruling of 2026-09-14 created a question it could not answer: the pre-correction incumbents
were fitted on data since found to be wrong, so RETAINING one is not a safe default, while
the anti-p-hacking prohibition still forbids PROMOTING a target with no PASS verdict. The
two cannot both be satisfied by a default. So the default is UNCHANGED -- a bare call still
promotes PASS targets only -- and a non-PASS promotion is possible ONLY through an explicit
recorded override carrying the owner's words and the date.

THE ASYMMETRY IS DELIBERATE AND BOTH DIRECTIONS ARE TESTED. A rule in one direction only
would either forbid the owner's ruling outright or wave it through unrecorded, so there are
two tests: a non-PASS promotion with NO override is REFUSED, and a non-PASS promotion WITH
one is accepted and records it.

ASCII only, no emoji (CLAUDE.md hard constraint).
"""

from __future__ import annotations

import inspect
import json
import tomllib
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pytest

from models import deploy_gate
from models.artifacts import (
    PREPROCESSING_FILENAME,
    PREPROCESSING_IS_MODEL_METADATA_KEY,
    load_model_artifact,
)
from scripts import run_phase33_gate as runner
from tests import phase33_gate_fixtures as fx
from tests import phase33_state
from tests.data_boundary import digest_file, digest_tree

COMMITTED_VERDICT_PATH = Path("config/phase33_gate_verdict.toml")
LIVE_MANIFEST_PATH = Path("artifacts") / "latest.json"

#: The closed verdict vocabulary, read from the runner rather than re-spelled.
VERDICT_STATES = frozenset(runner.VERDICT_STATES)


# ---------------------------------------------------------------------------
# Sandbox helpers
# ---------------------------------------------------------------------------


def _scoring(
    target: str,
    ids: list[str],
    *,
    delta: float,
    incumbent_ids: list[str] | None = None,
) -> runner.TargetScoring:
    """A ``TargetScoring`` whose paired delta decides the verdict.

    A POSITIVE delta passes the non-regression floor; a large NEGATIVE one fails it. The
    secondary metrics are pinned equal on both sides so the verdict turns on the CLV floor
    alone and a test reading FAIL knows which check produced it.
    """
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
        incumbent_scored=fx.scored_frame(
            target, incumbent_ids if incumbent_ids is not None else ids, value=value
        ),
        candidate_scored=fx.scored_frame(target, ids, value=value),
        gold=fx.gold_frame(
            target,
            sorted(set(ids) | set(incumbent_ids or [])),
        ),
        paired_delta=fx.paired_delta_frame(ids, delta=delta),
        extra={
            "windows": {"train": [2018], "hp_val": [2023], "holdout": [2024, 2025]},
            "windows_source": runner.WINDOWS_SOURCE,
            "window_report": "sandbox window report",
            "hyperparameter_search": "none",
            "resolved_hyperparameters": {"C": 1.0},
            "selected_feature_count": 3,
            "selected_feature_count_before": 3,
            "selected_weather_count_after": 1,
            "selected_weather_count_before": 0,
            "scorer_code_digest": "sandbox",
        },
    )


def _stage_one(
    tmp_path: Path,
    roots: fx.SandboxRoots,
    deltas: dict[str, float],
    *,
    targets: tuple[str, ...] = ("wp", "ats", "ou"),
    disjoint: tuple[str, ...] = (),
    now: str = "2026-09-15T00:00:00+00:00",
) -> runner.GateVerdictRecord:
    """Drive stage one over the sandbox, with per-target deltas choosing each verdict."""
    ids = fx.game_ids(120)
    other = [f"x{i:04d}" for i in range(120)]

    def _scorer(target: str, _staging: Path, _artifacts: Path) -> runner.TargetScoring:
        fx.stage_candidate_dirs(roots, (target,))
        return _scoring(
            target,
            ids,
            delta=deltas[target],
            incumbent_ids=other if target in disjoint else None,
        )

    return runner.stage_one_judge(
        scorer=_scorer,
        cfg=fx.gate_cfg(),
        targets=targets,
        artifacts_dir=roots.artifacts,
        staging_dir=roots.staging,
        verdict_record_path=tmp_path / "outputs" / "verdict.json",
        committed_verdict_path=tmp_path / "config" / "verdict.toml",
        min_free_bytes=0,
        now=now,
    )


def _manifest(root: Path) -> dict[str, Any]:
    return json.loads((root / "latest.json").read_text(encoding="utf-8"))


_OWNER_OVERRIDE = {
    "ruling": "promote-all-three",
    "ruled_on": "2026-09-14",
}


# ---------------------------------------------------------------------------
# (1) The committed record: three verdicts, the schema rule, the in-sample label
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def committed_verdicts() -> dict[str, Any]:
    """The committed generator-output half. READ ONLY; this module never writes it."""
    return tomllib.loads(COMMITTED_VERDICT_PATH.read_text(encoding="utf-8"))


class TestThreeVerdictsExistAgainstOneFrozenState:
    """R5: three verdicts, each naming PASS, FAIL or UNTESTABLE_REFUSAL."""

    def test_the_record_carries_exactly_the_three_gated_targets(
        self, committed_verdicts: dict[str, Any]
    ) -> None:
        assert sorted(committed_verdicts["verdicts"]) == ["ats", "ou", "wp"]

    def test_every_verdict_is_in_the_closed_vocabulary(
        self, committed_verdicts: dict[str, Any]
    ) -> None:
        states = {
            target: row["verdict"]
            for target, row in committed_verdicts["verdicts"].items()
        }
        assert set(states.values()) <= VERDICT_STATES, states

    def test_a_pass_or_fail_carries_a_statistic_and_a_p_value(
        self, committed_verdicts: dict[str, Any]
    ) -> None:
        """THE SCHEMA RULE, positive half. A claim about a measurement needs the number."""
        offenders = [
            target
            for target, row in committed_verdicts["verdicts"].items()
            if row["verdict"] in ("PASS", "FAIL")
            and (row.get("paired_statistic") is None or row.get("p_value") is None)
        ]
        assert offenders == [], offenders

    def test_an_untestable_refusal_may_carry_nulls_but_must_carry_a_reason(
        self, committed_verdicts: dict[str, Any]
    ) -> None:
        """THE SCHEMA RULE, asymmetric half.

        ``backtest.diagnose.clv_significance`` DELIBERATELY returns ``t=None, p=None``
        below ``MIN_CLV_SAMPLE``, and this phase's own success criterion says a
        zero-eligible-row target is REFUSED. Rejecting that as malformed would force a
        fabricated number into the record to make a legitimate refusal valid.
        """
        offenders = [
            target
            for target, row in committed_verdicts["verdicts"].items()
            if row["verdict"] == "UNTESTABLE_REFUSAL" and not row.get("reason")
        ]
        assert offenders == [], offenders

    def test_every_row_records_all_five_secondary_scalars(
        self, committed_verdicts: dict[str, Any]
    ) -> None:
        """FIVE, read from the constant. An earlier draft said four."""
        expected = len(deploy_gate.SECONDARY_SCALAR_NAMES)
        measured = {
            target: len(row.get("secondary_scalars") or {})
            for target, row in committed_verdicts["verdicts"].items()
        }
        assert set(measured.values()) == {expected}, (measured, expected)

    def test_no_feature_list_is_pinned_anywhere_in_the_record(
        self, committed_verdicts: dict[str, Any]
    ) -> None:
        """The shape the selection-procedure rule forbids."""
        pinned = [
            target
            for target, row in committed_verdicts["verdicts"].items()
            if row.get("feature_list") is not None
        ]
        assert pinned == [], pinned


class TestEveryVerdictIsLabelledInSample:
    """T-33-119: an in-sample verdict presented as a clean gate pass."""

    def test_the_record_header_carries_the_label(
        self, committed_verdicts: dict[str, Any]
    ) -> None:
        assert committed_verdicts.get("in_sample_label")

    def test_every_verdict_row_carries_the_label_and_a_window_report(
        self, committed_verdicts: dict[str, Any]
    ) -> None:
        for target, row in committed_verdicts["verdicts"].items():
            assert row.get("in_sample_label"), target
            assert row.get("window_report") is not None, target
            assert row["window_report"].strip(), target

    def test_the_committed_label_is_the_module_constant_and_has_not_drifted(
        self, committed_verdicts: dict[str, Any]
    ) -> None:
        """One label, three places: the constant, the record, and the state manifest."""
        assert committed_verdicts["in_sample_label"] == runner.IN_SAMPLE_VERDICT_LABEL
        assert phase33_state.GATE_VERDICT_SAMPLE_LABEL == runner.IN_SAMPLE_VERDICT_LABEL

    def test_the_label_says_it_is_not_an_out_of_sample_estimate(self) -> None:
        label = runner.IN_SAMPLE_VERDICT_LABEL.lower()
        assert "in-sample" in label
        assert "not" in label and "out-of-sample" in label

    def test_the_label_is_written_before_any_verdict_value_is_computed(self) -> None:
        """Structural, not observational: every return spreads the row that carries it.

        ``render_target_verdict`` builds ONE ``row`` dict containing the label and then
        spreads it into all four returns, including the three refusal returns that exit
        before any statistic exists. A label added on the success path only would leave a
        refusal unlabelled, which is precisely the row a reader is most likely to
        misread.
        """
        source = inspect.getsource(runner.render_target_verdict)
        label_at = source.find("in_sample_label")
        first_return = source.find("return {")
        assert 0 < label_at < first_return, (label_at, first_return)
        # EVERY return spreads the one row that carries the label -- the three refusal
        # returns included. A label added on the success path only would leave the
        # refusal rows unlabelled, and a refusal is the row a reader is most likely to
        # misread as "no result" rather than "no measurement was possible".
        assert source.count("**row,") == source.count("return {"), source.count(
            "**row,"
        )


class TestEveryWindowCameFromTheCommittedRule:
    """T-33-121: a window read out of a VOID pre-correction artifact's metadata."""

    def test_all_three_windows_equal_the_partition_rule_on_every_target(
        self, committed_verdicts: dict[str, Any]
    ) -> None:
        """The windows equal the rule AS IT STOOD when the verdict was produced.

        Re-pointed by Plan 33.2-18. The committed verdict is Phase 33's historical record
        and is never edited; the rule it ran under is recorded, also unedited, in
        ``tests.phase33_state.SEASON_PARTITION_AFTER`` (selection 2018-2022). The rule's
        second amendment (D33.2-14) then moved the selection window to 2002, so the verdict
        is compared with the rule of its own time, and the move is asserted rather than
        hidden: the LIVE partition's selection now differs from the verdict's train window.

        Was: ``want`` built from ``default_season_partition()``, the LIVE rule.
        """
        from conf.season_partition import default_season_partition

        recorded = phase33_state.SEASON_PARTITION_AFTER
        want = {
            "train": list(recorded["selection"]),
            "hp_val": list(recorded["hp_val"]),
            "holdout": list(recorded["holdout"]),
        }
        for target, row in committed_verdicts["verdicts"].items():
            assert {k: list(v) for k, v in row["windows"].items()} == want, target

        live = default_season_partition()
        assert list(live.selection) != want["train"], (
            "the live selection window equals the Phase-33 verdict's train window, but "
            "conf/season_partition.py's second amendment moved it to 2002"
        )
        assert list(live.hp_val) == want["hp_val"]
        assert list(live.holdout) == want["holdout"]

    def test_no_target_ran_a_hyperparameter_search(
        self, committed_verdicts: dict[str, Any]
    ) -> None:
        tuned = [
            target
            for target, row in committed_verdicts["verdicts"].items()
            if row.get("hyperparameter_search") not in (None, False, "none")
        ]
        assert tuned == [], tuned

    def test_the_no_tune_divergence_is_recorded_rather_than_discovered(self) -> None:
        reason = runner.NO_TUNE_DIVERGENCE_REASON
        assert "_build_train_argv" in reason
        assert "--no-tune" in reason
        assert "SPEC R5" in reason


# ---------------------------------------------------------------------------
# (2) The override: both directions
# ---------------------------------------------------------------------------


class TestANonPassPromotionRequiresARecordedOverride:
    """T-33-72: a promotion the verdict did not earn and nobody signed."""

    def test_a_non_pass_promotion_with_no_override_is_refused(
        self, tmp_path: Path
    ) -> None:
        roots = fx.sandbox_roots(tmp_path)
        _stage_one(tmp_path, roots, {"wp": 0.4, "ats": -0.4, "ou": -0.4})
        before = digest_file(roots.artifacts / "latest.json")

        with pytest.raises(runner.NonPassPromotionWithoutOverrideError) as excinfo:
            runner.stage_two_promote(
                verdict_record_path=tmp_path / "outputs" / "verdict.json",
                artifacts_dir=roots.artifacts,
                staging_dir=roots.staging,
                authorised_targets=("wp", "ats", "ou"),
            )

        message = str(excinfo.value)
        assert "ats" in message and "ou" in message
        assert "override" in message.lower()
        assert digest_file(roots.artifacts / "latest.json") == before, (
            "the refusal must leave the production swap surface untouched; a partial "
            "promotion is worse than none"
        )

    def test_a_non_pass_promotion_with_an_override_is_accepted_and_records_it(
        self, tmp_path: Path
    ) -> None:
        """The companion. A rule in one direction only would forbid the owner's ruling."""
        roots = fx.sandbox_roots(tmp_path)
        _stage_one(tmp_path, roots, {"wp": 0.4, "ats": -0.4, "ou": -0.4})

        promoted = runner.stage_two_promote(
            verdict_record_path=tmp_path / "outputs" / "verdict.json",
            artifacts_dir=roots.artifacts,
            staging_dir=roots.staging,
            authorised_targets=("wp", "ats", "ou"),
            overrides={"ats": _OWNER_OVERRIDE, "ou": _OWNER_OVERRIDE},
            # Stage two now appends each applied override to the committed TOML half
            # too (code review WR-13); point it at the sandbox half stage one wrote, or
            # the default would append to the real config/phase33_gate_verdict.toml.
            committed_verdict_path=tmp_path / "config" / "verdict.toml",
        )

        assert promoted == ("wp", "ats", "ou")
        manifest = _manifest(roots.artifacts)
        for target in ("wp", "ats", "ou"):
            assert manifest[target] == fx.CANDIDATE_VERSIONS[target]

    def test_an_override_missing_its_date_is_refused(self, tmp_path: Path) -> None:
        """An override with no date is an assertion nobody can place in time."""
        roots = fx.sandbox_roots(tmp_path)
        _stage_one(tmp_path, roots, {"wp": 0.4, "ats": -0.4, "ou": -0.4})

        with pytest.raises(runner.NonPassPromotionWithoutOverrideError):
            runner.stage_two_promote(
                verdict_record_path=tmp_path / "outputs" / "verdict.json",
                artifacts_dir=roots.artifacts,
                staging_dir=roots.staging,
                authorised_targets=("ats",),
                overrides={"ats": {"ruling": "promote-all-three", "ruled_on": ""}},
            )

    def test_the_default_is_unchanged_and_promotes_pass_targets_only(
        self, tmp_path: Path
    ) -> None:
        """Plan 33-08's behaviour is untouched by the override parameter existing."""
        roots = fx.sandbox_roots(tmp_path)
        _stage_one(tmp_path, roots, {"wp": 0.4, "ats": -0.4, "ou": -0.4})

        promoted = runner.stage_two_promote(
            verdict_record_path=tmp_path / "outputs" / "verdict.json",
            artifacts_dir=roots.artifacts,
            staging_dir=roots.staging,
        )

        assert promoted == ("wp",)
        manifest = _manifest(roots.artifacts)
        assert manifest["ats"] == fx.INCUMBENT_VERSIONS["ats"]
        assert manifest["ou"] == fx.INCUMBENT_VERSIONS["ou"]

    def test_a_pass_target_needs_no_override(self, tmp_path: Path) -> None:
        roots = fx.sandbox_roots(tmp_path)
        _stage_one(tmp_path, roots, {"wp": 0.4, "ats": 0.4, "ou": 0.4})

        promoted = runner.stage_two_promote(
            verdict_record_path=tmp_path / "outputs" / "verdict.json",
            artifacts_dir=roots.artifacts,
            staging_dir=roots.staging,
            authorised_targets=("wp", "ats", "ou"),
        )

        assert promoted == ("wp", "ats", "ou")


# ---------------------------------------------------------------------------
# (3) Copy before swap, and what a crash between them leaves behind
# ---------------------------------------------------------------------------


class TestCopyPrecedesEveryManifestWrite:
    """T-33-41d: a pointer at a staging-only version is a production outage."""

    def test_every_copy_precedes_every_manifest_write(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        roots = fx.sandbox_roots(tmp_path)
        _stage_one(tmp_path, roots, {"wp": 0.4, "ats": 0.4, "ou": 0.4})

        calls: list[str] = []
        real_copy = runner.promote_models._promote_artifact_dir
        real_update = runner.update_manifest

        def _copy(*args: Any, **kwargs: Any) -> Any:
            calls.append("copy")
            return real_copy(*args, **kwargs)

        def _update(*args: Any, **kwargs: Any) -> Any:
            calls.append("swap")
            return real_update(*args, **kwargs)

        monkeypatch.setattr(
            runner.promote_models, "_promote_artifact_dir", _copy, raising=True
        )
        monkeypatch.setattr(runner, "update_manifest", _update, raising=True)

        runner.stage_two_promote(
            verdict_record_path=tmp_path / "outputs" / "verdict.json",
            artifacts_dir=roots.artifacts,
            staging_dir=roots.staging,
            authorised_targets=("wp", "ats", "ou"),
        )

        assert calls == ["copy", "copy", "copy", "swap", "swap", "swap"], calls

    def test_every_promoted_target_resolves_after_the_swap(
        self, tmp_path: Path
    ) -> None:
        roots = fx.sandbox_roots(tmp_path)
        _stage_one(tmp_path, roots, {"wp": 0.4, "ats": 0.4, "ou": 0.4})
        runner.stage_two_promote(
            verdict_record_path=tmp_path / "outputs" / "verdict.json",
            artifacts_dir=roots.artifacts,
            staging_dir=roots.staging,
            authorised_targets=("wp", "ats", "ou"),
        )

        manifest = _manifest(roots.artifacts)
        for target in ("wp", "ats", "ou"):
            assert (roots.artifacts / manifest[target]).is_dir(), target

    def test_the_source_order_is_copy_then_swap(self) -> None:
        """Structural backstop for the call recorder."""
        source = inspect.getsource(runner.stage_two_promote)
        assert source.find("_promote_artifact_dir") < source.find("update_manifest")

    def test_a_crash_between_copy_and_swap_leaves_the_old_manifest_usable(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """The recovery property that makes the ordering worth having."""
        roots = fx.sandbox_roots(tmp_path)
        _stage_one(tmp_path, roots, {"wp": 0.4, "ats": 0.4, "ou": 0.4})
        before = json.loads(
            (roots.artifacts / "latest.json").read_text(encoding="utf-8")
        )

        def _explode(*_args: Any, **_kwargs: Any) -> None:
            msg = "simulated crash after the copies and before the swap"
            raise RuntimeError(msg)

        monkeypatch.setattr(runner, "update_manifest", _explode, raising=True)

        with pytest.raises(RuntimeError, match="simulated crash"):
            runner.stage_two_promote(
                verdict_record_path=tmp_path / "outputs" / "verdict.json",
                artifacts_dir=roots.artifacts,
                staging_dir=roots.staging,
                authorised_targets=("wp", "ats", "ou"),
            )

        after = json.loads(
            (roots.artifacts / "latest.json").read_text(encoding="utf-8")
        )
        assert after == before, "the OLD manifest must survive a crash unchanged"
        for target, version in after.items():
            assert (roots.artifacts / version).is_dir(), (target, version)


class TestTheManifestIsWrittenPerTargetAndNeverWholesale:
    """Pitfall 4, named in ``update_manifest``'s own docstring."""

    def test_update_manifest_is_called_once_per_promoted_target(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        roots = fx.sandbox_roots(tmp_path)
        # ONE passing target and two failing ones, so "once per PROMOTED target" is a
        # real count rather than a count that happens to equal the number of targets.
        _stage_one(tmp_path, roots, {"wp": 0.4, "ats": -0.4, "ou": -0.4})

        seen: list[str] = []
        real_update = runner.update_manifest

        def _update(target: str, *args: Any, **kwargs: Any) -> Any:
            seen.append(target)
            return real_update(target, *args, **kwargs)

        monkeypatch.setattr(runner, "update_manifest", _update, raising=True)

        runner.stage_two_promote(
            verdict_record_path=tmp_path / "outputs" / "verdict.json",
            artifacts_dir=roots.artifacts,
            staging_dir=roots.staging,
        )

        assert seen == ["wp"], seen

    def test_stage_two_never_writes_the_manifest_directly(self) -> None:
        source = inspect.getsource(runner.stage_two_promote)
        assert "json.dump" not in source
        assert "write_text" not in source

    def test_the_blend_pointer_is_byte_identical_before_and_after(
        self, tmp_path: Path
    ) -> None:
        """T-33-42: ``latest.json`` is not three independent slots."""
        roots = fx.sandbox_roots(tmp_path)
        _stage_one(tmp_path, roots, {"wp": 0.4, "ats": 0.4, "ou": 0.4})
        before = _manifest(roots.artifacts)["blend"]

        runner.stage_two_promote(
            verdict_record_path=tmp_path / "outputs" / "verdict.json",
            artifacts_dir=roots.artifacts,
            staging_dir=roots.staging,
            authorised_targets=("wp", "ats", "ou"),
        )

        assert _manifest(roots.artifacts)["blend"] == before

    def test_a_second_stage_two_run_leaves_the_manifest_identical(
        self, tmp_path: Path
    ) -> None:
        roots = fx.sandbox_roots(tmp_path)
        _stage_one(tmp_path, roots, {"wp": 0.4, "ats": 0.4, "ou": 0.4})
        record = tmp_path / "outputs" / "verdict.json"

        runner.stage_two_promote(
            verdict_record_path=record,
            artifacts_dir=roots.artifacts,
            staging_dir=roots.staging,
            authorised_targets=("wp", "ats", "ou"),
        )
        first = digest_file(roots.artifacts / "latest.json")
        runner.stage_two_promote(
            verdict_record_path=record,
            artifacts_dir=roots.artifacts,
            staging_dir=roots.staging,
            authorised_targets=("wp", "ats", "ou"),
        )

        assert digest_file(roots.artifacts / "latest.json") == first


class TestStageTwoCannotRunWithoutStageOne:
    """The prohibition is STRUCTURAL, not procedural."""

    def test_no_record_is_a_named_refusal(self, tmp_path: Path) -> None:
        roots = fx.sandbox_roots(tmp_path)
        with pytest.raises(runner.MissingStageOneVerdictError):
            runner.stage_two_promote(
                verdict_record_path=tmp_path / "outputs" / "absent.json",
                artifacts_dir=roots.artifacts,
                staging_dir=roots.staging,
                authorised_targets=("wp",),
            )

    def test_a_pass_record_with_a_missing_p_value_is_refused_not_promoted(
        self, tmp_path: Path
    ) -> None:
        roots = fx.sandbox_roots(tmp_path)
        record_path = tmp_path / "outputs" / "verdict.json"
        record_path.parent.mkdir(parents=True, exist_ok=True)
        record_path.write_text(
            json.dumps(
                {
                    "judge_version": "j",
                    "judge_code_digest": "d",
                    "rendered_at": "r",
                    "fix_cycle_allowance": 0,
                    "verdict_states": list(runner.VERDICT_STATES),
                    "secondary_scalar_names": list(deploy_gate.SECONDARY_SCALAR_NAMES),
                    "targets": {
                        "wp": {
                            "verdict": "PASS",
                            "paired_statistic": 3.0,
                            "p_value": None,
                            "candidate_version": fx.CANDIDATE_VERSIONS["wp"],
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

    def test_an_untestable_refusal_with_null_statistics_is_a_VALID_record(
        self, tmp_path: Path
    ) -> None:
        """The companion to the test above, and the asymmetry's whole point."""
        roots = fx.sandbox_roots(tmp_path)
        record_path = tmp_path / "outputs" / "verdict.json"
        record_path.parent.mkdir(parents=True, exist_ok=True)
        record_path.write_text(
            json.dumps(
                {
                    "judge_version": "j",
                    "judge_code_digest": "d",
                    "rendered_at": "r",
                    "fix_cycle_allowance": 0,
                    "verdict_states": list(runner.VERDICT_STATES),
                    "secondary_scalar_names": list(deploy_gate.SECONDARY_SCALAR_NAMES),
                    "targets": {
                        "wp": {
                            "verdict": "UNTESTABLE_REFUSAL",
                            "reason": runner.UntestableRefusalReason.ZERO_ELIGIBLE_ROWS.value,
                            "paired_statistic": None,
                            "p_value": None,
                            "candidate_version": fx.CANDIDATE_VERSIONS["wp"],
                        }
                    },
                }
            ),
            encoding="utf-8",
        )

        promoted = runner.stage_two_promote(
            verdict_record_path=record_path,
            artifacts_dir=roots.artifacts,
            staging_dir=roots.staging,
        )

        assert promoted == ()
        assert _manifest(roots.artifacts)["wp"] == fx.INCUMBENT_VERSIONS["wp"]


# ---------------------------------------------------------------------------
# (4) Stage one writes nothing, and the three SPEC R5 edges
# ---------------------------------------------------------------------------


class TestStageOneWritesNothingIntoProduction:
    """T-33-73, over the WHOLE tree and not merely the manifest."""

    def test_the_whole_production_root_is_byte_identical_across_stage_one(
        self, tmp_path: Path
    ) -> None:
        roots = fx.sandbox_roots(tmp_path)
        before = digest_tree(roots.artifacts)
        _stage_one(tmp_path, roots, {"wp": 0.4, "ats": -0.4, "ou": -0.4})
        assert digest_tree(roots.artifacts) == before

    def test_the_bracket_itself_can_fail(self, tmp_path: Path) -> None:
        """The control. A digest that never moves proves nothing about the one that did."""
        roots = fx.sandbox_roots(tmp_path)
        before = digest_tree(roots.artifacts)
        (roots.artifacts / "latest.json").write_text(
            '{"wp": "moved"}', encoding="utf-8"
        )
        assert digest_tree(roots.artifacts) != before


class TestTheThreeSpecR5Edges:
    """Each edge the SPEC names, carried by a test rather than by prose."""

    def test_a_p_value_exactly_at_alpha_resolves_per_the_strict_comparison(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """EQUALITY PASSES, because the committed operator is a STRICT ``<``.

        A delta array whose p-value lands EXACTLY on alpha cannot be constructed
        numerically, so the boundary is driven through the significance function the gate
        imports: mean strictly negative and p exactly alpha. Under
        ``not (mean < 0 and p < alpha)`` that is a PASS, and the source is asserted to
        carry the strict operator so the probe cannot agree with a rewritten rule.
        """
        alpha = deploy_gate.SIGNIFICANCE_ALPHA
        monkeypatch.setattr(
            deploy_gate,
            "clv_significance",
            lambda *_a, **_k: {"n": 100, "mean": -1.0, "t": -2.0, "p": alpha},
            raising=True,
        )

        assert deploy_gate.clv_non_regression_passes(np.zeros(100), alpha=alpha) is True
        assert "p" in inspect.getsource(deploy_gate.clv_non_regression_passes)
        assert 'sig["p"] < alpha' in inspect.getsource(
            deploy_gate.clv_non_regression_passes
        )

    def test_a_p_value_just_below_alpha_still_fails(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """The companion, so the equality test is not passing for a lazy reason."""
        alpha = deploy_gate.SIGNIFICANCE_ALPHA
        monkeypatch.setattr(
            deploy_gate,
            "clv_significance",
            lambda *_a, **_k: {"n": 100, "mean": -1.0, "t": -2.0, "p": alpha * 0.99},
            raising=True,
        )
        assert (
            deploy_gate.clv_non_regression_passes(np.zeros(100), alpha=alpha) is False
        )

    def test_zero_eligible_rows_is_refused_and_never_promoted_by_default(
        self, tmp_path: Path
    ) -> None:
        """A target with no paired rows is REFUSED, not FAILed with a fabricated number."""
        roots = fx.sandbox_roots(tmp_path)
        record = _stage_one(
            tmp_path,
            roots,
            {"wp": 0.4, "ats": 0.4, "ou": 0.4},
            disjoint=("ou",),
        )

        row = record.targets["ou"]
        assert row["verdict"] == "UNTESTABLE_REFUSAL"
        assert row["reason"] == runner.UntestableRefusalReason.ZERO_ELIGIBLE_ROWS.value
        assert row["paired_statistic"] is None
        assert row["p_value"] is None
        assert row["eligible_pairs"] == 0

        promoted = runner.stage_two_promote(
            verdict_record_path=tmp_path / "outputs" / "verdict.json",
            artifacts_dir=roots.artifacts,
            staging_dir=roots.staging,
        )
        assert "ou" not in promoted
        assert _manifest(roots.artifacts)["ou"] == fx.INCUMBENT_VERSIONS["ou"]

    def test_the_three_verdicts_are_invariant_to_target_order(
        self, tmp_path: Path
    ) -> None:
        deltas = {"wp": 0.4, "ats": -0.4, "ou": 0.05}
        forward = _stage_one(
            tmp_path / "a",
            fx.sandbox_roots(tmp_path / "a"),
            deltas,
            targets=("wp", "ats", "ou"),
        )
        reverse = _stage_one(
            tmp_path / "b",
            fx.sandbox_roots(tmp_path / "b"),
            deltas,
            targets=("ou", "ats", "wp"),
        )

        assert [t["verdict"] for t in forward.targets.values()] == [
            t["verdict"] for t in reverse.targets.values()
        ]
        assert list(forward.as_dict()["targets"]) == list(reverse.as_dict()["targets"])
        for target in ("wp", "ats", "ou"):
            assert (
                forward.targets[target]["paired_statistic"]
                == reverse.targets[target]["paired_statistic"]
            ), target


# ---------------------------------------------------------------------------
# (5) A split ensemble loads, and the WP preprocessing contract is served
# ---------------------------------------------------------------------------


def _synthetic_gold(n_features: int, seasons: tuple[int, ...]) -> pd.DataFrame:
    """A synthetic gold-shaped frame with a known signal in ``f0``."""
    rng = np.random.default_rng(3315004)
    rows: list[dict[str, Any]] = []
    for season in seasons:
        for i in range(24):
            values = rng.normal(size=n_features)
            row: dict[str, Any] = {
                "game_id": f"{season}_W{(i % 18) + 1:02d}_BUF@KC{i:02d}",
                "season": season,
                "week": (i % 18) + 1,
                "home_team": "KC",
                "away_team": "BUF",
            }
            for j, value in enumerate(values):
                row[f"f{j}"] = float(value)
            row["home_win"] = int(values[0] + 0.25 * values[1] > 0.0)
            row["home_margin"] = float(7.0 * values[0] + 2.0 * values[1])
            row["total_points"] = float(44.0 + 6.0 * values[0] - 3.0 * values[-1])
            rows.append(row)
    return pd.DataFrame(rows)


@pytest.fixture(scope="module")
def split_ensemble(tmp_path_factory: pytest.TempPathFactory) -> dict[str, Any]:
    """A sandbox roster mixing a NEW-CONTRACT WP with two differently-shaped siblings.

    This is the case the plan calls likely and the reviewers called risky: because each
    candidate RE-SELECTS rather than inheriting a feature list, a promoted model and a
    retained one can carry genuinely different feature sets. ATS is deliberately trained
    on a NARROWER frame so its selected list cannot coincide with the other two, and a
    test below asserts the three lists really do differ -- a mixture test on three
    identical rosters proves nothing.
    """
    from conf.season_partition import default_season_partition
    from models.train import train_target

    partition = default_season_partition()
    root = tmp_path_factory.mktemp("split_ensemble") / "artifacts"
    root.mkdir(parents=True, exist_ok=True)

    wide = _synthetic_gold(6, partition.final_fit)
    narrow = wide.drop(columns=["f4", "f5"])

    versions: dict[str, str] = {}
    for target, frame in (("wp", wide), ("ats", narrow), ("ou", wide)):
        result = train_target(
            target=target,
            features_df=frame,
            closing_odds_df=None,
            artifacts_dir=root,
            tune=False,
        )
        versions[target] = Path(result["artifact_path"]).name
    (root / "latest.json").write_text(json.dumps(versions, indent=2), encoding="utf-8")
    return {"root": root, "versions": versions, "gold": wide}


class TestASplitEnsembleLoadsAndPredicts:
    """Antigravity HIGH: a hybrid roster must load with no schema or feature mismatch."""

    def test_the_three_feature_lists_are_not_all_identical(
        self, split_ensemble: dict[str, Any]
    ) -> None:
        """Non-vacuity. A mixture test on three identical rosters proves nothing."""
        lists = {
            target: tuple(
                load_model_artifact(target, artifacts_dir=split_ensemble["root"])[
                    "feature_list"
                ]
            )
            for target in ("wp", "ats", "ou")
        }
        assert len(set(lists.values())) > 1, lists

    def test_the_prediction_path_produces_a_full_three_target_set(
        self, split_ensemble: dict[str, Any], monkeypatch: pytest.MonkeyPatch
    ) -> None:
        import scripts.generate_current_week_predictions as gen

        gold = split_ensemble["gold"]
        season = int(gold["season"].max())
        week = int(gold.loc[gold["season"] == season, "week"].min())
        slice_ = gold[(gold["season"] == season) & (gold["week"] == week)]
        assert not slice_.empty

        monkeypatch.setattr(
            gen,
            "load_gold_features",
            lambda _target, _season, _week: slice_.copy(),
            raising=True,
        )

        results = gen.run_predictions(split_ensemble["root"], season, week)

        assert sorted(results) == ["ats", "ou", "wp"]
        for target, column in (
            ("wp", "wp_prob"),
            ("ats", "ats_prediction"),
            ("ou", "ou_prediction"),
        ):
            frame = results[target]
            assert len(frame) == len(slice_), target
            assert column in frame.columns, target
            assert frame[column].notna().all(), target


class TestThePromotedWPArtifactShipsUnderThePreprocessingContract:
    """D33.1-R1: the transform and the estimator travel as ONE inseparable object."""

    def test_a_new_contract_wp_artifact_carries_preprocessing_on_disk(
        self, split_ensemble: dict[str, Any]
    ) -> None:
        directory = split_ensemble["root"] / split_ensemble["versions"]["wp"]
        metadata = json.loads((directory / "metadata.json").read_text(encoding="utf-8"))

        assert (directory / PREPROCESSING_FILENAME).is_file()
        assert metadata[PREPROCESSING_IS_MODEL_METADATA_KEY] is True

    def test_the_serving_path_resolves_the_transform_from_the_artifact(
        self, split_ensemble: dict[str, Any]
    ) -> None:
        """Not raw columns into ``predict_proba``: the Pipeline IS the estimator.

        ``load_model_artifact`` returns ONE object under both names when the artifact's
        own metadata records that the two files were serialized from the same estimator,
        so the two serving routes -- ``predict_games`` reads ``preprocessing`` for WP,
        everything else reads ``model`` -- cannot diverge across a duplicate.
        """
        from sklearn.pipeline import Pipeline

        artifact = load_model_artifact("wp", artifacts_dir=split_ensemble["root"])

        assert artifact["preprocessing"] is not None
        assert artifact["preprocessing"] is artifact["model"]
        assert isinstance(artifact["model"], Pipeline)
        assert artifact["model"].steps[-1][0] != "scaler"

    def test_a_legacy_artifact_is_still_served_and_never_refused(
        self, split_ensemble: dict[str, Any]
    ) -> None:
        """D33.1-R2: the contract binds artifacts saved UNDER it only."""
        artifact = load_model_artifact("ats", artifacts_dir=split_ensemble["root"])
        assert artifact["preprocessing"] is None
        assert artifact["model"] is not None


# ---------------------------------------------------------------------------
# (6) The production end state, against the recorded ruling
# ---------------------------------------------------------------------------


class TestThePromotedAndRetainedSetsAreRecordedAndDisjoint:
    """The end state Plan 33-16 and Plan 33-18 both read from one committed place."""

    def test_the_two_sets_partition_the_three_gated_targets(self) -> None:
        promoted = set(phase33_state.GATE_PROMOTED_TARGETS)
        retained = set(phase33_state.GATE_RETAINED_TARGETS)

        assert promoted | retained == {"wp", "ats", "ou"}
        assert promoted & retained == set()

    def test_an_untestable_target_is_kept_distinguishable_from_a_refusal_on_the_numbers(
        self,
    ) -> None:
        """A refusal for LACK OF EVIDENCE and one ON THE EVIDENCE are different findings."""
        untestable = set(phase33_state.GATE_UNTESTABLE_TARGETS)
        assert untestable <= set(phase33_state.GATE_RETAINED_TARGETS)
        assert untestable & set(phase33_state.GATE_PROMOTED_TARGETS) == set()

    def test_the_live_manifest_matches_the_recorded_end_state(self) -> None:
        """Production serves the LAST authorised swap's record, and nothing unrecorded.

        Re-pointed at the Phase 33 close-out. Plan 33.2-25's owner-accepted batched swap
        (2026-09-23) is now the most recent authorised write of ``artifacts/latest.json``,
        so the recorded end state is ``P332_25B_SWAP_ARTIFACT_IDS`` -- the same re-pointing
        ``test_weather_bridge_expiry`` and ``test_blend_revalidation_isolation`` made.
        ``POST_GATE_ARTIFACT_MANIFEST`` stays unedited as the record of what this gate
        installed, and the committed verdict TOML is not touched.

        Was: ``manifest == dict(phase33_state.POST_GATE_ARTIFACT_MANIFEST)``.

        RE-POINTED 2026-10-04 for WINDOWS row 19: the swap of the re-fits on the
        neutral-site Elo gold is now the most recent authorised write, so the end state is
        ``ROW19_SWAP_ARTIFACT_IDS``; ``P332_25B_SWAP_ARTIFACT_IDS`` stays as its record.
        """
        manifest = json.loads(LIVE_MANIFEST_PATH.read_text(encoding="utf-8"))
        assert manifest == dict(phase33_state.ROW19_SWAP_ARTIFACT_IDS)

    def test_every_live_pointer_resolves_to_a_directory_on_disk(self) -> None:
        manifest = json.loads(LIVE_MANIFEST_PATH.read_text(encoding="utf-8"))
        unresolvable = [
            (target, version)
            for target, version in manifest.items()
            if not (Path("artifacts") / version).is_dir()
        ]
        assert unresolvable == [], unresolvable

    def test_the_blend_pointer_is_the_unchanged_incumbent(self) -> None:
        """T-33-42 for THIS gate's promotion: it left the blend pointer where it found it.

        Re-pointed at the Phase 33 close-out to the gate's own recorded end state. The live
        blend pointer moved afterwards, deliberately, in Plan 33.2-25's authorised swap
        (``blend_20260923_212418``), so the live file can no longer witness what this
        promotion did; ``POST_GATE_ARTIFACT_MANIFEST`` was read off it right after the
        promotion and can. What production serves now is the test above's claim.

        Was: the LIVE ``artifacts/latest.json`` blend pointer.
        """
        recorded = dict(phase33_state.POST_GATE_ARTIFACT_MANIFEST)
        assert recorded["blend"] == dict(phase33_state.INCUMBENT_ARTIFACTS)["blend"]

    def test_every_non_pass_promotion_carries_an_owner_override(self) -> None:
        """The owner's ruling is recorded BESIDE the verdict, never inside it."""
        ruling = dict(phase33_state.GATE_NON_PASS_DISPOSITION_RULING)
        overrides = dict(ruling.get("override") or {})
        verdicts = dict(phase33_state.GATE_VERDICTS)

        unbacked = [
            target
            for target in phase33_state.GATE_PROMOTED_TARGETS
            if verdicts[target]["verdict"] != "PASS" and not overrides.get(target)
        ]
        assert unbacked == [], unbacked
        for target, override in overrides.items():
            assert override.get("ruling"), target
            assert override.get("ruled_on"), target

    def test_the_verdicts_were_not_softened_to_match_the_ruling(self) -> None:
        """THE POINT OF THE OWNER'S RULING: the measurement is published as measured.

        The committed record and the state manifest must still say FAIL for every target
        whose verdict was FAIL, however it was disposed of. An override recorded beside a
        verdict is a disclosure; a verdict re-read to look like a pass is the defect class
        this milestone exists to detect.
        """
        committed = tomllib.loads(COMMITTED_VERDICT_PATH.read_text(encoding="utf-8"))
        for target, row in dict(phase33_state.GATE_VERDICTS).items():
            assert committed["verdicts"][target]["verdict"] == row["verdict"], target
            assert committed["verdicts"][target]["paired_statistic"] == pytest.approx(
                row["paired_statistic"]
            ), target
