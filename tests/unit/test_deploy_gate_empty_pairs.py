"""Zero eligible paired rows is a REFUSAL, driven END TO END to a written record.

Phase 33, Plan 33-08 Task 3 (COLD-04, R5 empty edge, T-33-41 / T-33-41b / T-33-41c).

WHY THE VOCABULARY HAS A THIRD MEMBER
--------------------------------------
``backtest.diagnose.clv_significance`` DELIBERATELY returns ``t=None, p=None`` below
``MIN_CLV_SAMPLE`` (``backtest/diagnose.py:249-285``). A verdict schema that demanded
non-null statistics everywhere would reject that legitimate refusal as MALFORMED, and
the only way to make the record valid would be to fabricate a number. So the vocabulary
is ``PASS``, ``FAIL``, ``UNTESTABLE_REFUSAL``: the first two are claims about a
measurement and require statistics, the third is a claim that no measurement was
possible and requires a stated reason instead.

An ``UNTESTABLE_REFUSAL`` always RETAINS the incumbent. It is never a promotion, and a
target that could not be tested must never be promoted by default.

WHY END TO END AND NOT AT THE SIGNIFICANCE FUNCTION
-----------------------------------------------------
Simulating ``t is None`` by calling ``clv_significance`` on four numbers proves something
about scipy. What this phase needs proven is that an EMPTY ELIGIBILITY INDEX travels all
the way through the judge into a WRITTEN record whose verdict is ``UNTESTABLE_REFUSAL``,
whose statistics are null, whose reason is non-empty, and whose target's incumbent
pointer did not move. Every path between those two points is where the refusal could
quietly become a zero.

THE THREE PAIRING EDGES SHIP HERE TOO
--------------------------------------
Row-order invariance, a prediction missing on one side (with the excluded count landing
in the record) and a duplicate ``game_id`` (raising by name BEFORE any scoring) are the
three ways a paired comparison stops being paired without announcing it.

NO TEST HERE TOUCHES A PRODUCTION STORE. Every root comes from
``tests.phase33_gate_fixtures.sandbox_roots``, which is FAIL-CLOSED: it raises rather
than hand back a path inside the repository's real ``artifacts/`` tree.

Run this module:  uv run pytest tests/unit/test_deploy_gate_empty_pairs.py -q

ASCII only, no emoji (CLAUDE.md hard constraint).
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from models import deploy_gate as gate
from scripts import run_phase33_gate as runner
from tests import phase33_gate_fixtures as fx


def _scoring(
    target: str,
    *,
    incumbent_ids: list[str],
    candidate_ids: list[str],
    gold_ids: list[str] | None = None,
    delta: float = 0.4,
    with_delta: bool = True,
) -> runner.TargetScoring:
    """A ``TargetScoring`` with independently controllable id sets on each side."""
    gold_ids = incumbent_ids if gold_ids is None else gold_ids
    metrics = (
        {"accuracy": 0.70, "ece": 0.03, "brier_score": 0.20}
        if target == "wp"
        else {"mae": 0.0}
    )
    return runner.TargetScoring(
        target=target,
        candidate_version=fx.CANDIDATE_VERSIONS[target],
        incumbent_version=fx.INCUMBENT_VERSIONS[target],
        candidate_bundle=fx.candidate_bundle(target, **metrics),
        incumbent_scored=fx.scored_frame(target, incumbent_ids, value=0.5),
        candidate_scored=fx.scored_frame(target, candidate_ids, value=0.5),
        gold=fx.gold_frame(target, gold_ids),
        paired_delta=(
            fx.paired_delta_frame(
                sorted(set(incumbent_ids) & set(candidate_ids)), delta=delta
            )
            if with_delta
            else None
        ),
    )


def _run_stage_one(tmp_path: Path, scorings: dict[str, runner.TargetScoring]):
    """Drive stage one over a sandbox with a stubbed scorer; return (record, roots)."""
    roots = fx.sandbox_roots(tmp_path)
    record = runner.stage_one_judge(
        scorer=lambda target, _staging, _artifacts: scorings[target],
        cfg=fx.gate_cfg(),
        targets=tuple(scorings),
        artifacts_dir=roots.artifacts,
        staging_dir=roots.staging,
        verdict_record_path=tmp_path / "outputs" / "verdict.json",
        committed_verdict_path=tmp_path / "config" / "verdict.toml",
        min_free_bytes=0,
        now="2026-09-12T00:00:00+00:00",
    )
    return record, roots


class TestTheVerdictVocabularyIsClosedAndComplete:
    """Three members, and the schema rules are asymmetric on purpose."""

    def test_the_vocabulary_is_exactly_three_states(self) -> None:
        assert runner.VERDICT_STATES == ("PASS", "FAIL", "UNTESTABLE_REFUSAL")

    def test_it_matches_the_committed_manifest(self) -> None:
        from tests import phase33_state

        assert tuple(runner.VERDICT_STATES) == tuple(phase33_state.GATE_VERDICT_STATES)

    def test_a_verdict_outside_the_vocabulary_is_rejected_by_name(self) -> None:
        payload = _minimal_payload({"verdict": "PROBABLY", "reason": "x"})
        with pytest.raises(runner.MalformedVerdictRecordError) as excinfo:
            runner.validate_verdict_payload(payload)
        assert "PROBABLY" in str(excinfo.value)

    def test_a_pass_with_a_null_statistic_is_malformed(self) -> None:
        """PASS is a claim about a measurement, so it must carry one."""
        payload = _minimal_payload(
            {"verdict": "PASS", "paired_statistic": None, "p_value": 0.01}
        )
        with pytest.raises(runner.MalformedVerdictRecordError) as excinfo:
            runner.validate_verdict_payload(payload)
        assert "null paired" in str(excinfo.value)

    def test_an_untestable_refusal_without_a_reason_is_malformed(self) -> None:
        """A refusal that does not say why is indistinguishable from a missing result."""
        payload = _minimal_payload({"verdict": "UNTESTABLE_REFUSAL", "reason": "   "})
        with pytest.raises(runner.MalformedVerdictRecordError) as excinfo:
            runner.validate_verdict_payload(payload)
        assert "no reason" in str(excinfo.value)

    def test_an_untestable_refusal_with_null_statistics_is_LEGAL(self) -> None:
        """The whole reason the third state exists (T-33-41b)."""
        payload = _minimal_payload(
            {
                "verdict": "UNTESTABLE_REFUSAL",
                "reason": runner.UntestableRefusalReason.ZERO_ELIGIBLE_ROWS.value,
                "paired_statistic": None,
                "p_value": None,
            }
        )
        runner.validate_verdict_payload(payload)

    def test_every_declared_refusal_reason_is_a_non_empty_string(self) -> None:
        assert all(r.value.strip() for r in runner.UntestableRefusalReason)


def _minimal_payload(row_overrides: dict) -> dict:
    row = {
        "verdict": "PASS",
        "reason": "",
        "paired_statistic": -1.0,
        "p_value": 0.4,
        "candidate_version": "wp_x",
        "incumbent_version": "wp_y",
    }
    row.update(row_overrides)
    return {
        "judge_version": gate.JUDGE_VERSION,
        "judge_code_digest": gate.judge_code_digest(),
        "rendered_at": "2026-09-12T00:00:00+00:00",
        "fix_cycle_allowance": 0,
        "verdict_states": list(runner.VERDICT_STATES),
        "secondary_scalar_names": list(gate.SECONDARY_SCALAR_NAMES),
        "targets": {"wp": row},
    }


class TestAnEmptyEligibilityIndexIsRefusedEndToEnd:
    """From an empty index, through the judge, into a written record."""

    def test_the_written_record_says_untestable_refusal(self, tmp_path: Path) -> None:
        ids = fx.game_ids(40)
        record, _ = _run_stage_one(
            tmp_path,
            {"ou": _scoring("ou", incumbent_ids=ids[:15], candidate_ids=ids[25:])},
        )
        written = json.loads((tmp_path / "outputs" / "verdict.json").read_text("utf-8"))
        assert written["targets"]["ou"]["verdict"] == "UNTESTABLE_REFUSAL"
        assert record.targets["ou"]["verdict"] == "UNTESTABLE_REFUSAL"

    def test_its_statistics_are_null_rather_than_fabricated(
        self, tmp_path: Path
    ) -> None:
        ids = fx.game_ids(40)
        record, _ = _run_stage_one(
            tmp_path,
            {"ou": _scoring("ou", incumbent_ids=ids[:15], candidate_ids=ids[25:])},
        )
        row = record.targets["ou"]
        assert row["paired_statistic"] is None
        assert row["p_value"] is None
        assert row["n_paired"] == 0

    def test_its_reason_is_non_empty_and_names_the_zero_row_case(
        self, tmp_path: Path
    ) -> None:
        ids = fx.game_ids(40)
        record, _ = _run_stage_one(
            tmp_path,
            {"ou": _scoring("ou", incumbent_ids=ids[:15], candidate_ids=ids[25:])},
        )
        assert (
            record.targets["ou"]["reason"]
            == runner.UntestableRefusalReason.ZERO_ELIGIBLE_ROWS.value
        )

    def test_the_incumbent_pointer_is_unchanged(self, tmp_path: Path) -> None:
        """A refusal RETAINS. Asserted on the manifest, not on an intention."""
        ids = fx.game_ids(40)
        _, roots = _run_stage_one(
            tmp_path,
            {"ou": _scoring("ou", incumbent_ids=ids[:15], candidate_ids=ids[25:])},
        )
        manifest = json.loads((roots.artifacts / "latest.json").read_text("utf-8"))
        assert manifest == fx.INCUMBENT_VERSIONS

    def test_a_refusal_is_not_in_the_promotable_set(self, tmp_path: Path) -> None:
        ids = fx.game_ids(40)
        record, _ = _run_stage_one(
            tmp_path,
            {"ou": _scoring("ou", incumbent_ids=ids[:15], candidate_ids=ids[25:])},
        )
        assert record.passing_targets() == ()

    def test_a_sample_below_min_clv_sample_refuses_with_its_own_reason(
        self, tmp_path: Path
    ) -> None:
        """Non-empty but untestable is a DIFFERENT refusal from empty, and says so."""
        ids = fx.game_ids(5)
        record, _ = _run_stage_one(
            tmp_path, {"ou": _scoring("ou", incumbent_ids=ids, candidate_ids=ids)}
        )
        row = record.targets["ou"]
        assert row["verdict"] == "UNTESTABLE_REFUSAL"
        assert (
            row["reason"] == runner.UntestableRefusalReason.BELOW_MIN_CLV_SAMPLE.value
        )
        assert row["n_paired"] == 5

    def test_a_healthy_target_is_not_refused_so_the_refusals_are_not_vacuous(
        self, tmp_path: Path
    ) -> None:
        ids = fx.game_ids(120)
        record, _ = _run_stage_one(
            tmp_path, {"ou": _scoring("ou", incumbent_ids=ids, candidate_ids=ids)}
        )
        assert record.targets["ou"]["verdict"] == "PASS", record.targets["ou"]["reason"]


class TestTheThreePairingEdges:
    """Row order, a one-sided prediction, and a duplicate id."""

    def test_shuffling_both_sides_leaves_an_identical_verdict(
        self, tmp_path: Path
    ) -> None:
        ids = fx.game_ids(120)
        forward = _scoring("ou", incumbent_ids=ids, candidate_ids=ids)
        shuffled = runner.TargetScoring(
            target="ou",
            candidate_version=forward.candidate_version,
            incumbent_version=forward.incumbent_version,
            candidate_bundle=forward.candidate_bundle,
            incumbent_scored=forward.incumbent_scored.sample(
                frac=1.0, random_state=5
            ).reset_index(drop=True),
            candidate_scored=forward.candidate_scored.sample(
                frac=1.0, random_state=9
            ).reset_index(drop=True),
            gold=forward.gold.sample(frac=1.0, random_state=13).reset_index(drop=True),
            paired_delta=forward.paired_delta.sample(
                frac=1.0, random_state=17
            ).reset_index(drop=True),
        )
        a, _ = _run_stage_one(tmp_path / "a", {"ou": forward})
        b, _ = _run_stage_one(tmp_path / "b", {"ou": shuffled})
        assert a.as_dict() == b.as_dict()

    def test_a_prediction_missing_on_one_side_lands_in_the_record_as_a_count(
        self, tmp_path: Path
    ) -> None:
        """An asymmetric drop is VISIBLE, not absorbed into a smaller n."""
        ids = fx.game_ids(120)
        record, _ = _run_stage_one(
            tmp_path,
            {
                "ou": _scoring(
                    "ou", incumbent_ids=ids, candidate_ids=ids[4:], gold_ids=ids
                )
            },
        )
        eligibility = record.targets["ou"]["eligibility"]
        assert eligibility["excluded_incumbent_only"] == 4
        assert eligibility["excluded_candidate_only"] == 0
        assert eligibility["n_eligible"] == 116

    def test_a_duplicate_game_id_raises_by_name_before_any_scoring(
        self, tmp_path: Path
    ) -> None:
        import pandas as pd

        ids = fx.game_ids(30)
        base = _scoring("ou", incumbent_ids=ids, candidate_ids=ids)
        duped = runner.TargetScoring(
            target="ou",
            candidate_version=base.candidate_version,
            incumbent_version=base.incumbent_version,
            candidate_bundle=base.candidate_bundle,
            incumbent_scored=base.incumbent_scored,
            candidate_scored=pd.concat(
                [base.candidate_scored, base.candidate_scored.iloc[[7]]],
                ignore_index=True,
            ),
            gold=base.gold,
            paired_delta=base.paired_delta,
        )
        with pytest.raises(gate.DuplicateGameIdError) as excinfo:
            _run_stage_one(tmp_path, {"ou": duped})
        assert "g0007" in str(excinfo.value)
        assert "candidate" in str(excinfo.value)

    def test_a_target_with_no_paired_delta_at_all_refuses_rather_than_passing(
        self, tmp_path: Path
    ) -> None:
        ids = fx.game_ids(120)
        record, _ = _run_stage_one(
            tmp_path,
            {
                "ou": _scoring(
                    "ou", incumbent_ids=ids, candidate_ids=ids, with_delta=False
                )
            },
        )
        row = record.targets["ou"]
        assert row["verdict"] == "UNTESTABLE_REFUSAL"
        assert row["reason"] == runner.UntestableRefusalReason.NO_PAIRED_DELTA.value
