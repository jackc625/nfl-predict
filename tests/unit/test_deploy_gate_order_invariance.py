"""The three verdicts do not depend on the order the three targets were judged in.

Phase 33, Plan 33-08 Task 3 (COLD-04, R5 ordering edge).

WHAT COULD MAKE THIS FALSE, AND WHY IT IS WORTH ASSERTING
------------------------------------------------------------
Order-dependence in a judge is never designed in; it arrives through shared mutable
state. A cached comparator, a bundle mutated in place and reused, a module-level
accumulator, a verdict written and re-read mid-run -- each of those would make the
answer for O/U depend on whether WP ran first. None of them announces itself, and a
deploy gate whose verdict depends on iteration order is a gate that can be re-run until
it says what somebody wanted.

ALL SIX PERMUTATIONS, AND FULL RECORDS
---------------------------------------
Comparing only the PASS/FAIL labels would miss a paired statistic, an eligibility count
or a comparator scalar that moved -- which is exactly the class of drift that would
precede a label moving. So the assertion is on the WHOLE serialized record. The render
timestamp is pinned so the comparison is about the verdicts and not about the clock.

The three targets are deliberately given DIFFERENT outcomes -- one PASS, one FAIL, one
UNTESTABLE_REFUSAL -- because a permutation test over three identical verdicts would
pass on a judge that returned a constant.

NO TEST HERE TOUCHES A PRODUCTION STORE. Every root comes from the fail-closed
``tests.phase33_gate_fixtures.sandbox_roots``.

Run this module:  uv run pytest tests/unit/test_deploy_gate_order_invariance.py -q

ASCII only, no emoji (CLAUDE.md hard constraint).
"""

from __future__ import annotations

import itertools
from pathlib import Path

from scripts import run_phase33_gate as runner
from tests import phase33_gate_fixtures as fx

_PINNED_NOW = "2026-09-12T00:00:00+00:00"


def _three_distinct_scorings() -> dict[str, runner.TargetScoring]:
    """One PASS, one FAIL, one UNTESTABLE_REFUSAL.

    Three identical verdicts would let a constant-returning judge pass a permutation
    test, so the fixture makes the three outcomes different on purpose.
    """
    ids = fx.game_ids(120)
    short = fx.game_ids(40)

    scorings: dict[str, runner.TargetScoring] = {}

    # wp: PASSES -- candidate matches the live comparator on every secondary scalar and
    # the paired delta is positive.
    scorings["wp"] = runner.TargetScoring(
        target="wp",
        candidate_version=fx.CANDIDATE_VERSIONS["wp"],
        incumbent_version=fx.INCUMBENT_VERSIONS["wp"],
        candidate_bundle=fx.candidate_bundle(
            "wp", accuracy=1.0, ece=0.0, brier_score=0.0
        ),
        incumbent_scored=fx.scored_frame("wp", ids, value=0.5),
        candidate_scored=fx.scored_frame("wp", ids, value=0.5),
        gold=fx.gold_frame("wp", ids),
        paired_delta=fx.paired_delta_frame(ids, delta=0.4),
    )

    # ats: FAILS -- the paired CLV delta is significantly negative.
    scorings["ats"] = runner.TargetScoring(
        target="ats",
        candidate_version=fx.CANDIDATE_VERSIONS["ats"],
        incumbent_version=fx.INCUMBENT_VERSIONS["ats"],
        candidate_bundle=fx.candidate_bundle("ats", mae=0.0),
        incumbent_scored=fx.scored_frame("ats", ids, value=4.0),
        candidate_scored=fx.scored_frame("ats", ids, value=4.0),
        gold=fx.gold_frame("ats", ids),
        paired_delta=fx.paired_delta_frame(ids, delta=-0.4),
    )

    # ou: UNTESTABLE_REFUSAL -- the two sides share no game at all.
    scorings["ou"] = runner.TargetScoring(
        target="ou",
        candidate_version=fx.CANDIDATE_VERSIONS["ou"],
        incumbent_version=fx.INCUMBENT_VERSIONS["ou"],
        candidate_bundle=fx.candidate_bundle("ou", mae=0.0),
        incumbent_scored=fx.scored_frame("ou", short[:15], value=4.0),
        candidate_scored=fx.scored_frame("ou", short[25:], value=4.0),
        gold=fx.gold_frame("ou", short),
        paired_delta=fx.paired_delta_frame(short, delta=0.4),
    )
    return scorings


def _judge_in_order(tmp_path: Path, order: tuple[str, ...]) -> dict:
    scorings = _three_distinct_scorings()
    roots = fx.sandbox_roots(tmp_path)
    record = runner.stage_one_judge(
        scorer=lambda target, _s, _a: scorings[target],
        cfg=fx.gate_cfg(),
        targets=order,
        artifacts_dir=roots.artifacts,
        staging_dir=roots.staging,
        verdict_record_path=tmp_path / "outputs" / "verdict.json",
        committed_verdict_path=tmp_path / "config" / "verdict.toml",
        min_free_bytes=0,
        now=_PINNED_NOW,
    )
    return record.as_dict()


class TestTheVerdictsAreOrderInvariant:
    """All six permutations of (wp, ats, ou) produce byte-equal records."""

    def test_the_fixture_produces_three_different_verdicts(
        self, tmp_path: Path
    ) -> None:
        """Guards the guard: a constant-returning judge must not pass the test below."""
        record = _judge_in_order(tmp_path, ("wp", "ats", "ou"))
        verdicts = {t: record["targets"][t]["verdict"] for t in ("wp", "ats", "ou")}
        assert verdicts == {
            "wp": "PASS",
            "ats": "FAIL",
            "ou": "UNTESTABLE_REFUSAL",
        }, verdicts

    def test_all_six_orderings_produce_identical_full_records(
        self, tmp_path: Path
    ) -> None:
        """Full records, not just labels -- a moved statistic precedes a moved label."""
        permutations = list(itertools.permutations(("wp", "ats", "ou")))
        assert len(permutations) == 6
        records = [
            _judge_in_order(tmp_path / f"perm{i}", order)
            for i, order in enumerate(permutations)
        ]
        for order, record in zip(permutations[1:], records[1:]):
            assert record == records[0], f"ordering {order} produced a different record"

    def test_the_record_serializes_targets_in_canonical_order(
        self, tmp_path: Path
    ) -> None:
        """Whatever order the work ran in, the record reads the same way."""
        record = _judge_in_order(tmp_path, ("ou", "ats", "wp"))
        assert list(record["targets"]) == ["wp", "ats", "ou"]

    def test_the_promotable_set_is_order_invariant_too(self, tmp_path: Path) -> None:
        forward = _judge_in_order(tmp_path / "f", ("wp", "ats", "ou"))
        reverse = _judge_in_order(tmp_path / "r", ("ou", "ats", "wp"))
        passing = lambda rec: tuple(  # noqa: E731
            t for t, row in rec["targets"].items() if row["verdict"] == "PASS"
        )
        assert passing(forward) == passing(reverse) == ("wp",)
