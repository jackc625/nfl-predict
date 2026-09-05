"""The one-shot guards on the 2025 clean-split runner (Phase 31, plan 31-12; PROD-04, SPEC R3).

WHAT THIS MODULE IS ACTUALLY DEFENDING. The 2025 season is the only unburned split this project
has. It cannot be regenerated, bought or apologised for. Every assertion below is about the
machinery that stands between it and being spent twice, and each one is written to prove the
refusal is REACHABLE -- a guard that has only ever been observed passing is indistinguishable
from a guard that cannot fire.

Four refusal paths are exercised end to end:

1. An existing verdict artifact refuses the run BEFORE any chain runs -- proven with spies on
   the loaders and the selection seam showing ZERO calls, not merely by an exception type.
2. The run ledger is acquired with an EXCLUSIVE create, and a second acquisition raises rather
   than truncating.
3. The ledger exists on disk BEFORE the first hold-season read -- proven by call ORDER.
4. A crash injected between ledger acquisition and the artifact write leaves a ``failed``
   ledger and NO artifact, and the NEXT invocation refuses with zero hold reads. This is the
   exact window the artifact-existence check alone does not cover, and it is why the ledger
   exists at all.

NOTHING HERE READS 2025. Every run uses the disjoint rehearsal window (tune 2021-2023, hold
2024) over the synthetic fixture in ``tests/p31_synthetic_candidates.py``, which refuses to
generate 2025 at all.

ASCII only, no emoji (CLAUDE.md hard constraint).
"""

from __future__ import annotations

import ast
import json
import re
import tomllib
from pathlib import Path
from typing import Any

import pytest

import backtest.profitability_2025 as runner
from backtest.ats_ev_chain import FENCE_WINDOW_P31, FENCE_WINDOW_REHEARSAL
from backtest.diagnose import SIGNIFICANCE_ALPHA
from backtest.ev_chain_constants import REHEARSAL_PROXY_SPLIT, RUN_LEDGER_PATH
from tests.p31_synthetic_candidates import build_synthetic_candidates

REPO_ROOT = Path(__file__).resolve().parents[2]
RUNNER_PATH = REPO_ROOT / "backtest" / "profitability_2025.py"
RUNNER_SOURCE = RUNNER_PATH.read_text(encoding="utf-8")


@pytest.fixture(scope="module")
def synthetic_frames() -> dict[str, Any]:
    """The shared synthetic candidate frames (2018-2024; 2025 is unreachable)."""
    return build_synthetic_candidates()


def _paths(tmp_path: Path) -> dict[str, Path]:
    """The three throwaway run paths a rehearsal must name explicitly."""
    return {
        "verdict_toml_path": tmp_path / "verdict.toml",
        "verdict_json_path": tmp_path / "verdict.json",
        "ledger_path": tmp_path / "ledger.toml",
    }


class _Spy:
    """A counting wrapper that also records what was true on disk at each call."""

    def __init__(self, wrapped: Any, watch: Path | None = None) -> None:
        self._wrapped = wrapped
        self._watch = watch
        self.calls = 0
        self.watch_existed_at_call: list[bool] = []

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        self.calls += 1
        if self._watch is not None:
            self.watch_existed_at_call.append(self._watch.exists())
        return self._wrapped(*args, **kwargs)


@pytest.fixture(scope="module")
def completed_run(p31_rehearsal_run) -> dict[str, Any]:
    """ONE successful rehearsal run, shared with every other Phase-31 module that needs it.

    The run lives in ``tests/conftest.py`` at session scope: it costs a fifteen-cell tune
    sweep, a pooled three-target hold selection and three counterfactual passes, and four
    modules assert different properties of the same output.
    """
    return p31_rehearsal_run


class TestTheRunnerRefusesToOverwriteAVerdictBeforeItDoesAnyWork:
    """The artifact-existence refusal, validated at the same seam the Phase-30 judge uses."""

    def test_an_existing_verdict_refuses_before_any_chain_runs(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, synthetic_frames
    ) -> None:
        paths = _paths(tmp_path)
        paths["verdict_toml_path"].write_text(
            "# a verdict already lives here\n", encoding="utf-8"
        )

        tune_spy = _Spy(runner._load_tune_frames)
        hold_spy = _Spy(runner._load_hold_frames)
        select_spy = _Spy(runner._select)
        monkeypatch.setattr(runner, "_load_tune_frames", tune_spy)
        monkeypatch.setattr(runner, "_load_hold_frames", hold_spy)
        monkeypatch.setattr(runner, "_select", select_spy)

        with pytest.raises(runner.VerdictArtifactExistsError) as excinfo:
            runner.run_profitability_2025(
                FENCE_WINDOW_REHEARSAL,
                candidates_by_target=synthetic_frames,
                **paths,
            )

        assert "refusing to overwrite" in str(excinfo.value)
        assert "NO force flag" in str(excinfo.value)
        assert (tune_spy.calls, hold_spy.calls, select_spy.calls) == (0, 0, 0), (
            "the refusal arrived AFTER work had begun. A refusal that lands after a full "
            "walk-forward pass is a refusal nobody could afford to trust, which is why it is "
            "validated at the same seam backtest/group_gate.main validates its output path."
        )
        assert not paths["ledger_path"].exists(), (
            "the run ledger was acquired despite the refusal, so a refused invocation would "
            "block the next legitimate one."
        )

    def test_an_existing_run_record_also_refuses(
        self, tmp_path: Path, synthetic_frames
    ) -> None:
        """BOTH artifact forms are checked. Only checking one leaves half the record."""
        paths = _paths(tmp_path)
        paths["verdict_json_path"].write_text("{}\n", encoding="utf-8")
        with pytest.raises(runner.VerdictArtifactExistsError):
            runner.run_profitability_2025(
                FENCE_WINDOW_REHEARSAL,
                candidates_by_target=synthetic_frames,
                **paths,
            )

    def test_the_runner_refuses_to_write_under_the_data_tree(
        self, synthetic_frames
    ) -> None:
        """The HARD BOUNDARY: nothing this phase runs writes into the data lake."""
        with pytest.raises(ValueError, match="Refusing to write"):
            runner.run_profitability_2025(
                FENCE_WINDOW_REHEARSAL,
                candidates_by_target=synthetic_frames,
                verdict_toml_path=REPO_ROOT / "data" / "verdict.toml",
                verdict_json_path=REPO_ROOT / "outputs" / "p31" / "nope.json",
                ledger_path=REPO_ROOT / "outputs" / "p31" / "nope.toml",
            )


class TestThereIsNoEscapeHatch:
    """No flag, no argument and no environment variable can spend the split twice."""

    def test_the_parser_exposes_no_overwrite_force_or_clear_flag(self) -> None:
        parser = runner.build_parser()
        forbidden = re.compile(
            r"force|overwrite|no[-_]?check|skip|clear|reset|rearm|re[-_]arm|ignore|yes",
            re.IGNORECASE,
        )
        offending: list[str] = []
        for action in parser._actions:
            for option in action.option_strings:
                if forbidden.search(option):
                    offending.append(option)
            if action.help and forbidden.search(action.help):
                offending.append(f"help text of {action.option_strings or action.dest}")
        assert not offending, (
            "the CLI offers a way around the one-shot refusal: "
            f"{offending}. A flag that could spend the single-use split twice would be found "
            "and used eventually, so the design is that it does not exist."
        )

    def test_the_armed_entry_point_refuses_every_argument(self) -> None:
        """It takes no split, no window and no path, and says so by name."""
        with pytest.raises(runner.ProxySplitRefusedError, match="accepts NO arguments"):
            runner.run_armed_2025_verdict(split=REHEARSAL_PROXY_SPLIT)
        with pytest.raises(runner.ProxySplitRefusedError):
            runner.run_armed_2025_verdict(window=FENCE_WINDOW_REHEARSAL)
        with pytest.raises(runner.ProxySplitRefusedError):
            runner.run_armed_2025_verdict(verdict_toml_path="/tmp/x.toml")

    def test_the_frozen_window_cannot_be_redirected_onto_throwaway_paths(
        self, tmp_path: Path
    ) -> None:
        """Redirecting a binding run would let it escape the one-shot ledger entirely."""
        with pytest.raises(runner.ProxySplitRefusedError, match="not overridable"):
            runner.run_profitability_2025(
                FENCE_WINDOW_P31, ledger_path=tmp_path / "elsewhere.toml"
            )

    def test_a_rehearsal_must_name_its_own_paths_and_may_not_name_a_production_one(
        self, tmp_path: Path
    ) -> None:
        with pytest.raises(runner.ProxySplitRefusedError, match="must NAME"):
            runner.run_profitability_2025(FENCE_WINDOW_REHEARSAL)
        with pytest.raises(runner.ProxySplitRefusedError, match="PRODUCTION path"):
            runner.run_profitability_2025(
                FENCE_WINDOW_REHEARSAL,
                verdict_toml_path=tmp_path / "v.toml",
                verdict_json_path=tmp_path / "v.json",
                ledger_path=REPO_ROOT / RUN_LEDGER_PATH,
            )

    def test_an_unregistered_window_is_refused(self, tmp_path: Path) -> None:
        """ "Parameterised" must never become "arbitrary" for a leakage fence."""
        from backtest.ats_ev_chain import FenceWindow

        rogue = FenceWindow(
            tune_seasons=(2024, 2025),
            hold_seasons=(2025,),
            prior_residual_seasons=(2018,),
            threshold_window="tune_2024_2025",
            label="rogue",
            is_the_preregistered_rule=False,
        )
        with pytest.raises(runner.ProxySplitRefusedError, match="UNREGISTERED window"):
            runner.run_profitability_2025(
                rogue,
                verdict_toml_path=tmp_path / "v.toml",
                verdict_json_path=tmp_path / "v.json",
                ledger_path=tmp_path / "l.toml",
            )

    def test_mark_run_ledger_cannot_rearm(self, tmp_path: Path) -> None:
        ledger = tmp_path / "ledger.toml"
        runner.acquire_run_ledger(
            ledger, window=FENCE_WINDOW_REHEARSAL, preregistration_sha="deadbeef"
        )
        with pytest.raises(runner.RunLedgerError, match="refuses to write state"):
            runner.mark_run_ledger(ledger, "armed")


class TestTheDurableExclusiveRunLedger:
    """The crash-window control (REVIEW-ONESHOT), and every state that blocks."""

    def test_acquisition_is_an_exclusive_create_and_a_second_one_raises(
        self, tmp_path: Path
    ) -> None:
        ledger = tmp_path / "ledger.toml"
        first = runner.acquire_run_ledger(
            ledger, window=FENCE_WINDOW_REHEARSAL, preregistration_sha="abc123"
        )
        assert first["state"] == "started"
        before = ledger.read_bytes()

        with pytest.raises(runner.RunLedgerError) as excinfo:
            runner.acquire_run_ledger(
                ledger, window=FENCE_WINDOW_REHEARSAL, preregistration_sha="abc123"
            )
        assert first["attempt_id"] in str(excinfo.value)
        assert "'started'" in str(excinfo.value)
        assert ledger.read_bytes() == before, (
            "the second acquisition TRUNCATED the ledger. open(path, 'w') would do exactly "
            "that, and a truncated ledger is how a spent split becomes re-spendable."
        )

    def test_the_exclusive_create_uses_mode_x_in_source(self) -> None:
        """The mechanism, not just the behaviour: ``open(path, "x")`` and never ``"w"``."""
        tree = ast.parse(RUNNER_SOURCE)
        acquire = next(
            node
            for node in ast.walk(tree)
            if isinstance(node, ast.FunctionDef) and node.name == "acquire_run_ledger"
        )
        modes = [
            node.args[0].value
            for node in ast.walk(acquire)
            if isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and node.func.attr == "open"
            and node.args
            and isinstance(node.args[0], ast.Constant)
        ]
        assert "x" in modes, (
            "acquire_run_ledger does not open the ledger with the exclusive mode 'x'. Only 'x' "
            f"raises atomically instead of truncating; modes found: {modes}."
        )
        assert "w" not in modes

    def test_a_ledger_manually_set_to_started_blocks_the_next_invocation(
        self, tmp_path: Path, synthetic_frames
    ) -> None:
        paths = _paths(tmp_path)
        paths["ledger_path"].write_text(
            'state = "started"\nattempt_id = "manual-0001"\n'
            'started_at_utc = "2026-01-01T00:00:00+00:00"\n',
            encoding="utf-8",
        )
        with pytest.raises(runner.RunLedgerError) as excinfo:
            runner.run_profitability_2025(
                FENCE_WINDOW_REHEARSAL,
                candidates_by_target=synthetic_frames,
                **paths,
            )
        message = str(excinfo.value)
        assert "manual-0001" in message
        assert "'started'" in message
        assert "OWNER RULING" in message

    def test_an_unreadable_ledger_blocks_rather_than_reading_as_a_clean_slate(
        self, tmp_path: Path
    ) -> None:
        ledger = tmp_path / "ledger.toml"
        ledger.write_text("this is not = = valid toml\n", encoding="utf-8")
        with pytest.raises(runner.RunLedgerError, match="could not be read"):
            runner.acquire_run_ledger(
                ledger, window=FENCE_WINDOW_REHEARSAL, preregistration_sha="abc"
            )

    def test_the_binding_run_refuses_when_no_ledger_was_armed(
        self, tmp_path: Path
    ) -> None:
        """The accident guard: running the frozen window unarmed must not spend the split."""
        with pytest.raises(runner.RunLedgerError, match="never armed"):
            runner.acquire_run_ledger(
                tmp_path / "absent.toml",
                window=FENCE_WINDOW_P31,
                preregistration_sha="abc",
            )

    def test_arming_is_an_owner_act_that_cannot_be_repeated(
        self, tmp_path: Path
    ) -> None:
        ledger = tmp_path / "ledger.toml"
        with pytest.raises(runner.RunLedgerError, match="non-empty owner ruling"):
            runner.write_armed_ledger(ledger, owner_ruling="   ")

        armed = runner.write_armed_ledger(ledger, owner_ruling="CHECKPOINT 3 accepted")
        assert armed["state"] == "armed"
        with pytest.raises(runner.RunLedgerError, match="refusing to arm"):
            runner.write_armed_ledger(ledger, owner_ruling="again")

        started = runner.acquire_run_ledger(
            ledger, window=FENCE_WINDOW_P31, preregistration_sha="abc"
        )
        assert started["state"] == "started"
        assert started["prior_attempt_id"] == armed["attempt_id"]
        assert started["owner_ruling"] == "CHECKPOINT 3 accepted"

        with pytest.raises(runner.RunLedgerError):
            runner.acquire_run_ledger(
                ledger, window=FENCE_WINDOW_P31, preregistration_sha="abc"
            )


class TestTheLedgerIsHeldAcrossTheWholeReadToWriteWindow:
    """The ordering fact and the crash fact, both asserted rather than intended."""

    def test_the_ledger_exists_before_the_first_hold_read(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, synthetic_frames
    ) -> None:
        paths = _paths(tmp_path)
        hold_spy = _Spy(runner._load_hold_frames, watch=paths["ledger_path"])
        tune_spy = _Spy(runner._load_tune_frames, watch=paths["ledger_path"])
        monkeypatch.setattr(runner, "_load_hold_frames", hold_spy)
        monkeypatch.setattr(runner, "_load_tune_frames", tune_spy)

        runner.run_profitability_2025(
            FENCE_WINDOW_REHEARSAL,
            candidates_by_target=synthetic_frames,
            **paths,
        )

        assert hold_spy.calls == 1, (
            "the hold was read a number of times other than once; the ordering claim below "
            "only means something for a single, identifiable first read."
        )
        assert hold_spy.watch_existed_at_call[0] is True, (
            "the FIRST hold-season read happened while no run ledger existed on disk. A crash "
            "in that window would leave no artifact AND no record, and the next invocation "
            "would spend the single-use split again."
        )
        assert tune_spy.watch_existed_at_call[0] is True, (
            "the ledger is acquired before ANY candidate load, tune side included."
        )

    def test_a_crash_between_acquisition_and_the_write_leaves_a_failed_ledger(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, synthetic_frames
    ) -> None:
        """THE window the artifact-existence check alone does not cover."""
        paths = _paths(tmp_path)

        class _InjectedCrash(RuntimeError):
            """Stands in for a kill, a power loss or an unhandled exception mid-run."""

        def _crash(*_args: Any, **_kwargs: Any) -> None:
            assert paths["ledger_path"].exists()
            raise _InjectedCrash("simulated crash at the first hold read")

        monkeypatch.setattr(runner, "_load_hold_frames", _crash)
        with pytest.raises(_InjectedCrash):
            runner.run_profitability_2025(
                FENCE_WINDOW_REHEARSAL,
                candidates_by_target=synthetic_frames,
                **paths,
            )

        ledger = runner.read_run_ledger(paths["ledger_path"])
        assert ledger["state"] == "failed"
        assert ledger["failure_type"] == "_InjectedCrash"
        assert not paths["verdict_toml_path"].exists()
        assert not paths["verdict_json_path"].exists()

        # The NEXT invocation must refuse, and must refuse BEFORE reading the hold again.
        monkeypatch.undo()
        hold_spy = _Spy(runner._load_hold_frames)
        tune_spy = _Spy(runner._load_tune_frames)
        monkeypatch.setattr(runner, "_load_hold_frames", hold_spy)
        monkeypatch.setattr(runner, "_load_tune_frames", tune_spy)

        with pytest.raises(runner.RunLedgerError) as excinfo:
            runner.run_profitability_2025(
                FENCE_WINDOW_REHEARSAL,
                candidates_by_target=synthetic_frames,
                **paths,
            )
        message = str(excinfo.value)
        assert ledger["attempt_id"] in message
        assert "'failed'" in message
        assert "OWNER RULING" in message
        assert (hold_spy.calls, tune_spy.calls) == (0, 0), (
            "the blocked second invocation still read data. A 'failed' attempt is exactly the "
            "case where the hold may already have been read once."
        )


class TestTheRunnerComposesOnlyCanonicalPrimitives:
    """It adds the correction and the verdict, and re-derives nothing else."""

    def test_alpha_is_the_one_imported_alpha(self) -> None:
        assert runner.ALPHA is SIGNIFICANCE_ALPHA

    def test_the_module_declares_no_local_statistic(self) -> None:
        """No second alpha, no second significance test, no second resampler, no second devig."""
        tree = ast.parse(RUNNER_SOURCE)
        defined = {
            node.name
            for node in ast.walk(tree)
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
        }
        forbidden = re.compile(
            r"bootstrap_(ci|replicates|resample)|resample|ttest|significance|devig|"
            r"american_to_|calibrated_p_|flat_roi_from",
            re.IGNORECASE,
        )
        offending = sorted(name for name in defined if forbidden.search(name))
        assert not offending, (
            f"the judge defines its own {offending}. Every one of those is an existing "
            "canonical primitive; a second implementation is a second answer."
        )
        imported = {
            alias.name
            for node in ast.walk(tree)
            if isinstance(node, ast.ImportFrom)
            for alias in node.names
        }
        for expected in (
            "roi_ci_and_p",
            "devig",
            "clv_significance",
            "estimate_prior_season_bias",
            "fit_frozen_residual_sd",
            "american_to_payout",
            "ALPHA",
        ):
            assert expected in imported, (
                f"{expected} is not imported; the judge must consume it rather than restate it."
            )

    def test_the_roi_p_value_comes_from_the_preregistered_module(self) -> None:
        tree = ast.parse(RUNNER_SOURCE)
        sources = {
            node.module
            for node in ast.walk(tree)
            if isinstance(node, ast.ImportFrom)
            and any(alias.name == "roi_ci_and_p" for alias in node.names)
        }
        assert sources == {"backtest.roi_significance"}

    def test_the_module_never_reaches_the_phase27_fence(self) -> None:
        assert "_assert_fit_window" not in RUNNER_SOURCE, (
            "the Phase-27 fence helper READS the Phase-27 window, so a Phase-31 runner reusing "
            "it inherits 2023-2024 as the hold SILENTLY -- the fence still passes and its "
            "report names the wrong seasons (D31-14, T-31-19)."
        )

    def test_the_preregistration_commit_is_resolved_and_never_transcribed(self) -> None:
        resolved = runner.preregistration_commit()
        assert re.fullmatch(r"[0-9a-f]{40}", resolved), resolved
        assert not re.search(r"['\"][0-9a-f]{40}['\"]", RUNNER_SOURCE), (
            "a 40-character SHA literal appears in the runner's source. The anchor is RESOLVED "
            "from git precisely so there is no second, silently divergable copy of the one "
            "fact SPEC R3's ancestry assertion rests on."
        )


class TestBothArtifactFormsComeFromOneRenderer:
    """A committed TOML and a machine-readable record that cannot disagree about a number."""

    def test_the_run_produced_both_forms(self, completed_run) -> None:
        assert completed_run["verdict_toml_path"].exists()
        assert completed_run["verdict_json_path"].exists()

    def test_the_numeric_fields_of_the_two_forms_are_exactly_equal(
        self, completed_run
    ) -> None:
        parsed = tomllib.loads(
            completed_run["verdict_toml_path"].read_text(encoding="utf-8")
        )
        record = json.loads(
            completed_run["verdict_json_path"].read_text(encoding="utf-8")
        )
        fields = record["verdict_fields"]

        compared = 0
        for target, target_fields in fields["targets"].items():
            for name, text in target_fields.items():
                if name not in parsed["targets"][target]:
                    continue
                value = parsed["targets"][target][name]
                if isinstance(value, float):
                    assert runner.render_float(value) == text, (
                        f"targets.{target}.{name} disagrees between the two artifact forms: "
                        f"TOML {value!r} renders as {runner.render_float(value)!r} but the run "
                        f"record carries {text!r}."
                    )
                    compared += 1
        assert compared > 0, (
            "no float field was compared, so this assertion proved nothing about the two "
            "forms agreeing."
        )

    def test_the_ledger_is_completed_and_names_both_artifacts(
        self, completed_run
    ) -> None:
        ledger = runner.read_run_ledger(completed_run["ledger_path"])
        assert ledger["state"] == "completed"
        assert ledger["verdict_artifact"].endswith("verdict.toml")
        assert ledger["verdict_run_record"].endswith("verdict.json")

    def test_the_fence_report_names_the_rehearsal_window_and_never_2025(
        self, completed_run
    ) -> None:
        reports = completed_run["result"]["fence_reports"]
        assert set(reports) == {"wp", "ats", "ou"}
        for target, report in reports.items():
            assert report["hold_seasons"] == [2024], target
            assert report["tune_seasons"] == [2021, 2022, 2023], target
            assert report["fence_held"] is True
            assert 2025 not in report["hold_seasons"] + report["tune_seasons"]
            assert report["window_is_the_preregistered_rule"] is False

    def test_no_2025_figure_appears_in_a_rehearsal_artifact(
        self, completed_run
    ) -> None:
        """The rehearsal result is DISCARDED; it must not even mention the unspent split."""
        text = completed_run["verdict_toml_path"].read_text(encoding="utf-8")
        assert "2025" not in text.split("# =====")[-1], (
            "a rehearsal artifact names 2025. The rehearsal is not the pre-registered rule and "
            "its result is never reported, never published and never compared."
        )
