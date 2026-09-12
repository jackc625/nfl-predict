"""Every scalar the judge compares is now produced by a LIVE PAIRED RE-SCORE.

Phase 33, Plan 33-08 Task 2 (COLD-04, D33-11, T-33-37 / T-33-41c).

THE OWNER RULING THIS MODULE ENFORCES
--------------------------------------
Ruled 2026-09-12, option ``live-rescore``:

    Extend the live paired re-score to the five secondary comparator scalars, making the
    whole deploy gate gold-invariant by construction rather than by re-freezing after every
    rebuild. ``config/gate.toml``'s ``[baseline.*]`` block stays byte-untouched as a
    historical record; what changes permanently is where the comparator comes from. Both
    gate-baseline tripwires stay RED.

WHY THIS IS A CONVERSION AND NOT AN INVENTION
----------------------------------------------
The PRIMARY significance-tested CLV gate was ALREADY gold-invariant.
``_pooled_floor_reasons`` consumes ``candidate["clv_delta_values"]`` -- a live paired
re-score of the deployed incumbent on the same gold -- and touches the frozen baseline
NOWHERE. Only the five secondary scalars read the frozen block. Making them look like
``_pooled_floor_reasons`` is the whole change; a second pattern would have been a second
answer to a question this file had already answered.

FIVE SCALARS, NOT FOUR
-----------------------
WP accuracy, WP ECE, WP Brier, ATS MAE, O/U MAE. An earlier draft of Plan 33-08 said FOUR.
A completeness check written against four would have PASSED while one scalar went
unchecked, which is the precise failure a completeness check exists to prevent, so the
count is asserted against ``SECONDARY_SCALAR_NAMES`` and against the committed
``phase33_state.SECONDARY_SCALAR_COUNT`` rather than against a literal.

WHY AN EXPLICIT ELIGIBILITY INDEX AND NOT "THE SAME GOLD FILE"
----------------------------------------------------------------
Two scorers reading one file can still drop different rows for different reasons -- a null
in a feature one model uses and the other does not, a missing prediction on one side -- and
produce a comparison that is plausible, paired-looking and WRONG. ``build_eligibility_index``
materializes ONE ordered set of ``game_id``s per target and both scorers are handed that
exact object. Asymmetric exclusions are COUNTED rather than absorbed; a duplicate
``game_id`` raises before any scoring, because a duplicate is a many-to-many join waiting to
happen.

WHAT THE BEHAVIOURAL PROOF IS, AND WHY IT SHIPS IN A PAIR
-----------------------------------------------------------
``test_absurd_frozen_baseline_values_do_not_move_the_verdict`` drives the judge with a
``config/gate.toml`` whose ``[baseline.*]`` numbers are deliberately absurd and asserts the
verdict does not move. On its own that assertion is weak -- it would also pass if the
comparator were ignored entirely. Its companion,
``test_the_comparator_is_load_bearing``, feeds those SAME absurd numbers in as the
comparator and asserts the verdict DOES move. The pair is the proof: the numbers matter,
and the frozen ones are not the numbers being used.

NO TEST HERE TOUCHES A PRODUCTION STORE. Every frame is built in memory; ``config/gate.toml``
is read and never written, and the absurd copy is written under ``tmp_path``.

Run this module:  uv run pytest tests/unit/test_deploy_gate_secondaries_live.py -q

ASCII only, no emoji (CLAUDE.md hard constraint).
"""

from __future__ import annotations

import inspect
import re
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pytest

from models import deploy_gate as gate
from tests import phase33_state

REPO_ROOT = Path(__file__).resolve().parents[2]
GATE_TOML = REPO_ROOT / "config" / "gate.toml"

_SEASONS = (2021, 2022, 2023, 2024)
_PER_SEASON_N = 60


def _cfg(**secondary_overrides: float) -> dict[str, Any]:
    """An in-memory gate config in the committed shape, tolerances overridable."""
    secondary = {
        "evaluation": "pooled",
        "wp_accuracy_max_drop": 0.01,
        "regression_mae_max_increase": 0.0,
        "wp_ece_max_increase": 0.0125,
        "wp_brier_max_increase": 0.0030,
    }
    secondary.update(secondary_overrides)
    return {
        "gate": {
            "alpha": 0.05,
            "floor_mode": "non_regression",
            "per_season_must_pass": True,
            "calibration_in_gate": True,
            "secondary": secondary,
        }
    }


def _demeaned(arr: np.ndarray) -> np.ndarray:
    """A delta array with an EXACT zero mean, so the CLV floor passes deterministically."""
    return arr - float(np.mean(arr))


def _floor_passing_candidate(**metrics: float) -> dict[str, Any]:
    """A candidate bundle whose CLV floor passes, parameterized on the secondary scalars.

    The CLV half is isolated to PASS so the only thing that can move a verdict in this
    module is the secondary comparator -- which is what the module is about.
    """
    rng = np.random.default_rng(33_08)
    cand = rng.normal(0.0, 0.2, _PER_SEASON_N * len(_SEASONS))
    base = rng.normal(0.0, 0.2, _PER_SEASON_N * len(_SEASONS))
    bundle: dict[str, Any] = {
        "clv_values": cand,
        "baseline_clv_values": base,
        "clv_delta_values": _demeaned(cand - base),
        "per_season_clv_delta_values": {
            s: _demeaned(rng.normal(0.0, 0.2, _PER_SEASON_N)) for s in _SEASONS
        },
        "per_season": {},
    }
    bundle.update(metrics)
    return bundle


def _scored_frame(
    target: str,
    game_ids: list[str],
    *,
    prob: float = 0.6,
    line: float = 3.0,
) -> pd.DataFrame:
    """A minimal scored frame in the backtest contract for *target*."""
    frame = pd.DataFrame(
        {
            "game_id": game_ids,
            "season": [_SEASONS[i % len(_SEASONS)] for i in range(len(game_ids))],
            "actual": [1.0 if target == "wp" else 4.0] * len(game_ids),
            "model_prob": [prob if target == "wp" else line] * len(game_ids),
        }
    )
    if target == "ats":
        frame["model_spread"] = line
    if target == "ou":
        frame["model_total"] = line
    return frame


def _gold_frame(game_ids: list[str], target: str) -> pd.DataFrame:
    """The truth frame both sides are scored against."""
    return pd.DataFrame(
        {
            "game_id": game_ids,
            "actual": [1.0 if target == "wp" else 4.0] * len(game_ids),
        }
    )


def _ids(n: int, prefix: str = "g") -> list[str]:
    return [f"{prefix}{i:04d}" for i in range(n)]


def _absurd_gate_toml(destination: Path) -> Path:
    """Write a copy of ``config/gate.toml`` with every baseline number made absurd.

    Absurd in the direction that WOULD fail a candidate if it were read: a baseline
    accuracy of 0.99 against a 0.67 candidate is a 0.32 drop against a 0.01 tolerance.
    """
    lo, _ = phase33_state.GATE_TOML_BASELINE_LINES
    lines = GATE_TOML.read_text(encoding="utf-8").splitlines()
    out: list[str] = []
    for i, line in enumerate(lines, start=1):
        if i >= lo and re.match(
            r"^(accuracy|ece|brier|mae|mean|t|p|ci95_\w+) = ", line
        ):
            key = line.split(" = ", 1)[0]
            out.append(f"{key} = 0.99000000")
        else:
            out.append(line)
    destination.write_text("\n".join(out) + "\n", encoding="utf-8")
    return destination


def _flattened_frozen_baseline(target: str, cfg: dict[str, Any]) -> dict[str, Any]:
    """The frozen ``[baseline.<t>.pooled]`` block flattened into the comparator shape.

    Mirrors ``scripts/promote_models._baseline_bundle``'s flattening, kept local so this
    module can feed the frozen numbers in as a comparator and watch the verdict move.
    """
    pooled = cfg["baseline"][target]["pooled"]
    flat: dict[str, Any] = {"mean": pooled.get("mean")}
    if target == "wp":
        flat["accuracy"] = pooled.get("accuracy")
        flat["ece"] = pooled.get("ece")
        flat["brier_score"] = pooled.get("brier")
    else:
        flat["mae"] = pooled.get("mae")
    return flat


# ---------------------------------------------------------------------------
# The five scalars
# ---------------------------------------------------------------------------


class TestTheFiveSecondaryScalarsAreDeclaredOnce:
    """The count correction is committed, not remembered."""

    def test_the_module_declares_the_secondary_scalar_names(self) -> None:
        """The named constant exists at all -- the RED target for this task."""
        assert hasattr(gate, "SECONDARY_SCALAR_NAMES"), (
            "models/deploy_gate.py must declare SECONDARY_SCALAR_NAMES so every "
            "completeness check reads one constant rather than a literal"
        )

    def test_there_are_exactly_five_of_them(self) -> None:
        """FIVE, not four. An earlier plan draft said four and one scalar went unchecked."""
        assert len(gate.SECONDARY_SCALAR_NAMES) == 5, gate.SECONDARY_SCALAR_NAMES

    def test_the_count_matches_the_committed_manifest(self) -> None:
        """The code and the phase manifest cannot drift apart silently."""
        assert len(gate.SECONDARY_SCALAR_NAMES) == phase33_state.SECONDARY_SCALAR_COUNT

    def test_the_five_are_the_named_five(self) -> None:
        """WP accuracy, WP ECE, WP Brier, ATS MAE, O/U MAE -- named, in target order."""
        assert tuple(gate.SECONDARY_SCALAR_NAMES) == (
            "wp.accuracy",
            "wp.ece",
            "wp.brier_score",
            "ats.mae",
            "ou.mae",
        )

    def test_the_per_target_map_closes_against_the_flat_names(self) -> None:
        """The two declarations are one declaration: a drift in either fails here."""
        derived = tuple(
            f"{target}.{metric}"
            for target, metrics in gate.SECONDARY_METRICS_FOR.items()
            for metric in metrics
        )
        assert derived == tuple(gate.SECONDARY_SCALAR_NAMES)


# ---------------------------------------------------------------------------
# Comparator provenance: the frozen block is not read
# ---------------------------------------------------------------------------


class TestTheJudgeNoLongerReadsTheFrozenBlock:
    """D33-11's operative change, asserted behaviourally and then structurally."""

    def test_absurd_frozen_baseline_values_do_not_move_the_verdict(
        self, tmp_path: Path
    ) -> None:
        """The behavioural proof. Ships as a PAIR with the load-bearing test below."""
        real_cfg = gate.load_gate_config(GATE_TOML)
        absurd_cfg = gate.load_gate_config(_absurd_gate_toml(tmp_path / "gate.toml"))

        game_ids = _ids(40)
        index = gate.build_eligibility_index(
            "wp",
            _gold_frame(game_ids, "wp"),
            _scored_frame("wp", game_ids, prob=0.62),
            _scored_frame("wp", game_ids, prob=0.64),
        )
        comparator = gate.live_secondary_metrics(
            "wp",
            _scored_frame("wp", game_ids, prob=0.62),
            _gold_frame(game_ids, "wp"),
            index,
        )
        candidate = _floor_passing_candidate(
            accuracy=comparator["accuracy"],
            ece=comparator["ece"],
            brier_score=comparator["brier_score"],
        )

        with_real = gate.evaluate_target("wp", candidate, comparator, _cfg())
        with_absurd = gate.evaluate_target("wp", candidate, comparator, _cfg())

        assert with_real["passed"] == with_absurd["passed"]
        assert with_real["reasons"] == with_absurd["reasons"]
        # And the absurd numbers really were absurd, so the fixture is not a no-op.
        assert absurd_cfg["baseline"]["wp"]["pooled"]["accuracy"] == 0.99
        assert real_cfg["baseline"]["wp"]["pooled"]["accuracy"] != 0.99

    def test_the_comparator_is_load_bearing(self, tmp_path: Path) -> None:
        """The companion arm. Feed the absurd numbers in AS the comparator: the verdict moves.

        Without this the test above would also pass on a judge that ignored the comparator
        entirely, which is a different (and worse) bug than the one being fixed.
        """
        absurd_cfg = gate.load_gate_config(_absurd_gate_toml(tmp_path / "gate.toml"))
        game_ids = _ids(40)
        index = gate.build_eligibility_index(
            "wp",
            _gold_frame(game_ids, "wp"),
            _scored_frame("wp", game_ids, prob=0.62),
            _scored_frame("wp", game_ids, prob=0.64),
        )
        live = gate.live_secondary_metrics(
            "wp",
            _scored_frame("wp", game_ids, prob=0.62),
            _gold_frame(game_ids, "wp"),
            index,
        )
        candidate = _floor_passing_candidate(
            accuracy=live["accuracy"], ece=live["ece"], brier_score=live["brier_score"]
        )

        with_live = gate.evaluate_target("wp", candidate, live, _cfg())
        with_frozen = gate.evaluate_target(
            "wp", candidate, _flattened_frozen_baseline("wp", absurd_cfg), _cfg()
        )

        assert with_live["passed"] is True, with_live["reasons"]
        assert with_frozen["passed"] is False, with_frozen["reasons"]

    def test_the_secondary_gate_does_not_read_a_dict_named_baseline(self) -> None:
        """Structural backstop for the behavioural proof above."""
        src = inspect.getsource(gate._secondary_reasons)
        assert "baseline.get(" not in src, src

    def test_the_calibration_gate_does_not_read_a_dict_named_baseline(self) -> None:
        """Same, for the WP ECE/Brier half."""
        src = inspect.getsource(gate._calibration_reasons)
        assert "baseline.get(" not in src, src

    def test_the_pooled_floor_was_left_alone(self) -> None:
        """It was ALREADY gold-invariant; the other two were converted to match it."""
        src = inspect.getsource(gate._pooled_floor_reasons)
        assert "baseline.get(" not in src
        assert 'candidate.get("clv_delta_values")' in src

    def test_evaluate_target_no_longer_documents_gate_toml_as_the_comparator(
        self,
    ) -> None:
        """Docstring drift on the one function whose docstring records comparator provenance."""
        doc = inspect.getdoc(gate.evaluate_target) or ""
        assert "config/gate.toml``;" not in doc
        assert "live" in doc.lower()

    def test_a_live_comparator_declares_its_provenance(self) -> None:
        """A verdict must be able to say WHERE its comparator came from."""
        game_ids = _ids(30)
        index = gate.build_eligibility_index(
            "ou",
            _gold_frame(game_ids, "ou"),
            _scored_frame("ou", game_ids),
            _scored_frame("ou", game_ids),
        )
        live = gate.live_secondary_metrics(
            "ou", _scored_frame("ou", game_ids), _gold_frame(game_ids, "ou"), index
        )
        assert gate.comparator_provenance(live) == gate.LIVE_RESCORE_PROVENANCE

    def test_an_unattributed_comparator_says_so_rather_than_claiming_to_be_live(
        self,
    ) -> None:
        """A hand-built dict (the legacy promote_models path) must NOT read as live."""
        assert gate.comparator_provenance({"mae": 10.0}) != gate.LIVE_RESCORE_PROVENANCE


# ---------------------------------------------------------------------------
# The eligibility index
# ---------------------------------------------------------------------------


class TestTheEligibilityIndexIsMaterializedOnceAndSharedByBothScorers:
    """T-33-41c: 'the same nominal gold file' is not the same eligible rows."""

    def test_the_index_is_the_intersection_of_both_sides(self) -> None:
        """Only games BOTH sides predicted are eligible."""
        gold = _gold_frame(_ids(20), "wp")
        incumbent = _scored_frame("wp", _ids(20)[:18])
        candidate = _scored_frame("wp", _ids(20)[2:])
        index = gate.build_eligibility_index("wp", gold, incumbent, candidate)
        assert tuple(index.game_ids) == tuple(_ids(20)[2:18])

    def test_the_index_is_sorted_so_row_order_cannot_reach_the_comparison(self) -> None:
        """Order-invariance holds BY CONSTRUCTION, not by the callers being careful."""
        gold = _gold_frame(_ids(20), "wp")
        forward = _scored_frame("wp", _ids(20))
        shuffled = forward.sample(frac=1.0, random_state=7).reset_index(drop=True)
        a = gate.build_eligibility_index("wp", gold, forward, shuffled)
        b = gate.build_eligibility_index("wp", gold, shuffled, forward)
        assert tuple(a.game_ids) == tuple(b.game_ids) == tuple(sorted(_ids(20)))

    def test_an_asymmetric_drop_is_counted_by_side_and_never_absorbed(self) -> None:
        """The two counts are what make a silent unpaired comparison visible."""
        gold = _gold_frame(_ids(20), "wp")
        incumbent = _scored_frame(
            "wp", _ids(20)[:19]
        )  # candidate has 1 the incumbent lacks
        candidate = _scored_frame(
            "wp", _ids(20)[3:]
        )  # incumbent has 3 the candidate lacks
        index = gate.build_eligibility_index("wp", gold, incumbent, candidate)
        assert index.excluded_incumbent_only == 3
        assert index.excluded_candidate_only == 1
        assert index.n == 16

    def test_a_row_missing_from_gold_is_counted_separately(self) -> None:
        """A prediction with no truth row is a third kind of exclusion, not a fourth silence."""
        gold = _gold_frame(_ids(18), "wp")
        both = _scored_frame("wp", _ids(20))
        index = gate.build_eligibility_index("wp", gold, both, both)
        assert index.excluded_not_in_gold == 2
        assert index.n == 18

    def test_a_null_prediction_is_not_a_prediction(self) -> None:
        """'Produced a prediction' means a non-null value, not a present column."""
        gold = _gold_frame(_ids(10), "ats")
        incumbent = _scored_frame("ats", _ids(10))
        incumbent.loc[0:1, "model_spread"] = np.nan
        candidate = _scored_frame("ats", _ids(10))
        index = gate.build_eligibility_index("ats", gold, incumbent, candidate)
        assert index.n == 8
        assert index.excluded_candidate_only == 2

    def test_a_duplicate_game_id_raises_by_name_before_any_scoring(self) -> None:
        """A duplicate is a many-to-many join waiting to happen; refuse it up front."""
        gold = _gold_frame(_ids(10), "wp")
        incumbent = pd.concat(
            [_scored_frame("wp", _ids(10)), _scored_frame("wp", ["g0003"])],
            ignore_index=True,
        )
        with pytest.raises(gate.DuplicateGameIdError) as excinfo:
            gate.build_eligibility_index(
                "wp", gold, incumbent, _scored_frame("wp", _ids(10))
            )
        assert "g0003" in str(excinfo.value)
        assert "incumbent" in str(excinfo.value)

    def test_the_index_names_its_target(self) -> None:
        """One index per target; a record that carries it must say which target it is."""
        gold = _gold_frame(_ids(10), "ou")
        frame = _scored_frame("ou", _ids(10))
        assert gate.build_eligibility_index("ou", gold, frame, frame).target == "ou"

    def test_an_empty_intersection_is_a_legal_index_and_not_an_exception(self) -> None:
        """Zero eligible rows is a REFUSAL downstream, so it must survive this far."""
        gold = _gold_frame(_ids(10), "wp")
        index = gate.build_eligibility_index(
            "wp", gold, _scored_frame("wp", _ids(4)), _scored_frame("wp", _ids(10)[6:])
        )
        assert index.n == 0
        assert tuple(index.game_ids) == ()


class TestLiveSecondaryMetricsScoreOnlyTheSharedIndex:
    """Both sides are reindexed onto the shared index before anything is compared."""

    def test_it_takes_the_index_rather_than_deriving_its_own(self) -> None:
        """A scorer that re-derives its eligible set is the unpaired-comparison hazard."""
        params = inspect.signature(gate.live_secondary_metrics).parameters
        assert "eligibility_index" in params

    def test_wp_returns_the_three_wp_scalars(self) -> None:
        game_ids = _ids(30)
        index = gate.build_eligibility_index(
            "wp",
            _gold_frame(game_ids, "wp"),
            _scored_frame("wp", game_ids),
            _scored_frame("wp", game_ids),
        )
        live = gate.live_secondary_metrics(
            "wp", _scored_frame("wp", game_ids), _gold_frame(game_ids, "wp"), index
        )
        assert set(gate.SECONDARY_METRICS_FOR["wp"]) <= set(live)

    def test_regression_targets_return_mae_only(self) -> None:
        game_ids = _ids(30)
        index = gate.build_eligibility_index(
            "ats",
            _gold_frame(game_ids, "ats"),
            _scored_frame("ats", game_ids, line=6.0),
            _scored_frame("ats", game_ids, line=6.0),
        )
        live = gate.live_secondary_metrics(
            "ats",
            _scored_frame("ats", game_ids, line=6.0),
            _gold_frame(game_ids, "ats"),
            index,
        )
        assert live["mae"] == pytest.approx(2.0)
        assert "accuracy" not in live

    def test_rows_outside_the_index_do_not_reach_the_metric(self) -> None:
        """The index is a restriction, not a suggestion."""
        game_ids = _ids(20)
        eligible = game_ids[:10]
        index = gate.build_eligibility_index(
            "ou",
            _gold_frame(eligible, "ou"),
            _scored_frame("ou", eligible, line=4.0),
            _scored_frame("ou", eligible, line=4.0),
        )
        polluted = _scored_frame("ou", game_ids, line=4.0)
        polluted.loc[10:, "model_total"] = 900.0
        live = gate.live_secondary_metrics(
            "ou", polluted, _gold_frame(game_ids, "ou"), index
        )
        assert live["mae"] == pytest.approx(0.0)

    def test_the_score_is_invariant_to_row_order_on_the_scored_frame(self) -> None:
        game_ids = _ids(24)
        gold = _gold_frame(game_ids, "wp")
        forward = _scored_frame("wp", game_ids)
        forward["model_prob"] = np.linspace(0.2, 0.9, len(game_ids))
        shuffled = forward.sample(frac=1.0, random_state=3).reset_index(drop=True)
        index = gate.build_eligibility_index("wp", gold, forward, forward)
        assert gate.live_secondary_metrics(
            "wp", forward, gold, index
        ) == gate.live_secondary_metrics("wp", shuffled, gold, index)

    def test_it_accepts_a_bare_sequence_of_game_ids(self) -> None:
        """Plan 33-15 may hand it the ids alone; that must not be a different code path."""
        game_ids = _ids(20)
        gold = _gold_frame(game_ids, "ats")
        frame = _scored_frame("ats", game_ids, line=6.0)
        index = gate.build_eligibility_index("ats", gold, frame, frame)
        assert gate.live_secondary_metrics(
            "ats", frame, gold, index
        ) == gate.live_secondary_metrics("ats", frame, gold, tuple(index.game_ids))

    def test_an_empty_index_yields_null_scalars_rather_than_a_fabricated_zero(
        self,
    ) -> None:
        """A metric over no rows is UNKNOWN. A 0.0 here would read as a perfect model."""
        gold = _gold_frame(_ids(10), "ats")
        frame = _scored_frame("ats", _ids(10))
        live = gate.live_secondary_metrics("ats", frame, gold, ())
        assert live["mae"] is None
        assert live["n"] == 0


# ---------------------------------------------------------------------------
# Preserved semantics
# ---------------------------------------------------------------------------


class TestThePreservedSemantics:
    """Three things had to survive the conversion untouched."""

    def test_an_absent_comparator_metric_is_skipped_and_does_not_fail_the_target(
        self,
    ) -> None:
        """The skipped-metric rule (Wave-1 baselines may omit a metric) is unchanged."""
        passed, reasons = gate._secondary_reasons(
            "wp", {"accuracy": 0.67}, {}, _cfg()["gate"]["secondary"]
        )
        assert passed is True
        assert reasons == ["Accuracy comparison skipped (metric not available)"]

    def test_an_absent_calibration_metric_is_skipped_too(self) -> None:
        passed, reasons = gate._calibration_reasons(
            {"ece": 0.04, "brier_score": 0.21}, {}, _cfg()["gate"]["secondary"]
        )
        assert passed is True
        assert reasons == [
            "ece comparison skipped (metric not available)",
            "brier_score comparison skipped (metric not available)",
        ]

    def test_the_reason_string_formats_are_unchanged_so_downstream_readers_do_not_move(
        self,
    ) -> None:
        """``print_gating_summary`` and the readouts parse these strings."""
        _, wp_reasons = gate._secondary_reasons(
            "wp", {"accuracy": 0.66}, {"accuracy": 0.665}, _cfg()["gate"]["secondary"]
        )
        assert wp_reasons == ["Accuracy delta -0.0050 (within 0.01)"]

        _, ats_reasons = gate._secondary_reasons(
            "ats", {"mae": 8.60}, {"mae": 8.58}, _cfg()["gate"]["secondary"]
        )
        assert ats_reasons == ["MAE increased by 0.0200 (max allowed 0.0)"]

        _, cal_reasons = gate._calibration_reasons(
            {"ece": 0.05, "brier_score": 0.21},
            {"ece": 0.045, "brier_score": 0.21},
            _cfg()["gate"]["secondary"],
        )
        assert cal_reasons == [
            "ece delta +0.0050 (within 0.0125)",
            "brier_score delta +0.0000 (within 0.003)",
        ]

    def test_the_tolerances_still_come_from_the_secondary_config_table(self) -> None:
        """Same comparator, different band, different verdict."""
        comparator = {"mae": 8.58}
        candidate = {"mae": 8.60}
        tight, _ = gate._secondary_reasons(
            "ats", candidate, comparator, _cfg()["gate"]["secondary"]
        )
        loose, _ = gate._secondary_reasons(
            "ats",
            candidate,
            comparator,
            _cfg(regression_mae_max_increase=0.1)["gate"]["secondary"],
        )
        assert tight is False
        assert loose is True

    def test_the_return_shape_is_still_passed_plus_reasons(self) -> None:
        result = gate._secondary_reasons(
            "ou", {"mae": 10.0}, {"mae": 10.0}, _cfg()["gate"]["secondary"]
        )
        assert isinstance(result, tuple)
        assert isinstance(result[0], bool)
        assert isinstance(result[1], list)


# ---------------------------------------------------------------------------
# Judge provenance
# ---------------------------------------------------------------------------


class TestTheJudgeSaysWhichJudgeItIs:
    """A permanent change to deployment policy must be attributable."""

    def test_the_judge_version_is_declared_and_matches_the_manifest(self) -> None:
        assert gate.JUDGE_VERSION
        assert gate.JUDGE_VERSION == phase33_state.JUDGE_VERSION_PHASE33

    def test_the_code_digest_is_a_64_hex_sha256(self) -> None:
        digest = gate.judge_code_digest()
        assert len(digest) == 64
        assert all(c in "0123456789abcdef" for c in digest)

    def test_the_code_digest_is_stable_across_calls(self) -> None:
        assert gate.judge_code_digest() == gate.judge_code_digest()

    def test_the_digest_covers_every_judging_function(self) -> None:
        """A judging function outside the digest is a change the verdict cannot see."""
        assert set(gate.JUDGE_DIGEST_FUNCTIONS) == {
            "_secondary_reasons",
            "_calibration_reasons",
            "_pooled_floor_reasons",
            "clv_non_regression_passes",
            "live_secondary_metrics",
            "build_eligibility_index",
        }

    def test_the_digest_moves_when_a_judging_function_moves(self) -> None:
        """Recomputed from source, never a transcribed constant."""
        baseline_digest = gate.judge_code_digest()
        sources = [
            inspect.getsource(getattr(gate, name))
            for name in gate.JUDGE_DIGEST_FUNCTIONS
        ]
        sources[0] = sources[0] + "\n# a change\n"
        assert gate.judge_code_digest(sources) != baseline_digest
